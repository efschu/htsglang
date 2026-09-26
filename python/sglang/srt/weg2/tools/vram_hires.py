#!/usr/bin/env python3
"""VRAM-Sonde mit 5-ms-Aufloesung je Karte und je Prozess (fnFL2 H55).

Nutzer 24.09. 18:20Z: "kannst du keine sonde bauen die das wirklich mitschneidet
wie viel vram tatsaechlich belegt ist? und zwar so hochaufgeloest, dass wir
daraus auch wirklich schluesse ziehen koennen?" -- 18:22Z: hochaufgeloest meint
das ZEITINTERVALL; die 1-s-Sicht (``nvidia-smi dmon``) ist genau das, was nicht
mehr reicht.

WARUM DMON NICHT REICHT. ``dmon -d 1`` liest einmal je ~1,1 s (x162: 17:05:10
und 17:05:18 fehlen ganz) einen Momentwert. Eine Chunk-Transiente von 3,7 GiB,
die 300 ms steht, faellt mit ~70 % Wahrscheinlichkeit zwischen zwei Lesungen;
x149 starb bei 338 MiB Kopfraum, und keine dmon-Zeile hatte je weniger als
~1 GiB frei gezeigt. Diese Sonde liest NVML alle ``--period-ms`` (Standard 5)
und schreibt JEDEN Wechsel -- ein Maximum, das zwei Abtastungen lang stand, ist
im Rohstrom.

KEIN CUDA-KONTEXT. Nur NVML (pynvml, im venv als nvidia-ml-py 13.x): kein
cuInit, kein Kontext, kein Byte VRAM, keine SM-Zeit. Der Preis ist ein
RM-ioctl je Karte und Abtastung; der Kopf der CSV nennt ihn GEMESSEN
(``mean_call_us``/``p99_call_us`` aus der Kalibrierung) und die ERREICHTE Rate
(``rate_hz``). Liegt der Tick-Preis ueber ``--budget`` x Periode, hebt die Sonde
die Periode selbst an und sagt es im Kopf (``period_raised_from=``). Sie laeuft
mit ``nice`` 10, damit sie dem PLE-Gather der Raenge keine CPU nimmt.

AUSGABE (zwei Dateien, beide mit demselben Kopf):

``<out>.raw.csv`` -- ROHSTROM, Standard AN (``--no-raw`` schaltet ihn ab).
    ``t_ms,key,used_mib`` -- ``t_ms`` monoton in ms seit t0 (``t0_unix_ms`` im
    Kopf; ``# sync``-Zeilen alle 60 s halten die Wanduhr nach). Geschrieben wird
    jeder WERTWECHSEL einer Reihe plus ein Herzschlag je Reihe und Sekunde --
    verlustfrei fuer Maximum/Minimum (der Wert steht bis zur naechsten Zeile)
    und ~50x kleiner als jede Abtastung einzeln. Reihen: ``c<idx>`` = Karte
    (fb used), ``c<idx>:<pid>`` = Prozess auf der Karte, ``c<idx>:other`` =
    Karte minus Summe der Prozesse (Treiber/fremd, bei jeder Prozess-Abtastung).
    ``# proc``-Zeilen nennen je neuem Prozess die Rolle (``P:PP0``, ``D:TP1``,
    ``?:<comm>``) und den Boot-Tag, ``# gap``-Zeilen jede verspaetete Abtastung
    (> 3 Perioden), ``# stat``-Zeilen alle 10 s die laufende Rate.

``<out>`` (``.csv``) -- 1-s-Eimer, an der Unix-Sekunde ausgerichtet (so liegen
    sie auf den ``[.. HH:MM:SS PP0]``-Zeilen der P/D-Logs):
    ``utc,sec_unix,key,role,min_mib,max_mib,last_mib,free_min_mib,n`` --
    ``free_min_mib`` = kleinste Restluft der Karte in dieser Sekunde (nur fuer
    ``c<idx>``), ``n`` = Abtastungen im Eimer.

FREI. ``free = total - offset - used`` mit ``offset`` je Karte aus der
Kalibrierung (``total - used - free`` der NVML-Lesung, i.e. die vom Treiber
reservierten MiB, 5090 ~518, 3080 ~425) -- dieselbe Zahl, die cudaMemGetInfo
als frei meldet und die der Planer als ``card free`` liest.

LEBENSDAUER (H52-Lehre: 107 verwaiste Logger). Drei Riegel, jeder allein genug:
PR_SET_PDEATHSIG (stirbt der Elternprozess, kommt SIGTERM), ``--owner-pattern``
(argv-Regex des Boots; erscheint er nicht binnen ``--owner-grace-s`` oder ist er
``--owner-gone-s`` lang weg: Ende), ``--max-s`` (hartes Ende). SIGTERM/SIGINT
schreiben den letzten Eimer und eine ``# end``-Zeile.

    python3 vram_hires.py --out /spinning/evidence-665-f1/vram_hires_TAG.csv \\
        --tag TAG --owner-pattern 'sglang\\.srt\\.weg2\\.(launcher|front) .*--tag TAG( |$)'

Leser: ``vram_hires_report.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

FORMAT_VERSION = 1
MIB = 1 << 20

#: Standard-Periode (Nutzer 18:22Z: mindestens 10 ms, Ziel 5 ms).
DEFAULT_PERIOD_MS = 5.0
#: Prozess-Abtastung (nvmlDeviceGetComputeRunningProcesses ist teurer als
#: MemoryInfo; die Kartensumme laeuft trotzdem mit der vollen Rate).
DEFAULT_PROC_EVERY_MS = 10.0
#: Anteil der Periode, den ein Tick hoechstens kosten darf, bevor die Sonde die
#: Periode anhebt.
DEFAULT_BUDGET = 0.30
#: Herzschlag je Reihe im Rohstrom (ms), damit eine stille Reihe lesbar bleibt.
KEEPALIVE_MS = 1000
#: Abstand der ``# stat``- und ``# sync``-Zeilen.
STAT_EVERY_MS = 10_000
SYNC_EVERY_MS = 60_000
#: Kalibrierung: so viele Ticks vor dem Kopf.
CALIB_TICKS = 100
#: Eine Abtastung, die mehr als so viele Perioden zu spaet kommt, ist ein Gap.
GAP_PERIODS = 3.0

BOOT_JSON_DIR = "/spinning/gpu-arb/weg2"


# --------------------------------------------------------------------------
# NVML-Quelle (echte und Attrappe teilen diese Schnittstelle)
# --------------------------------------------------------------------------


class NvmlSource:
    """Duenne Huelle um pynvml. ``mem(i)`` -> (used, free, total) in Bytes,
    ``procs(i)`` -> [(pid, used_bytes or None)]. Kein CUDA, nur NVML."""

    def __init__(self, indices: Sequence[int]):
        import pynvml  # nvidia-ml-py; bewusst erst hier (Tests laufen ohne)

        self._n = pynvml
        pynvml.nvmlInit()
        self._h = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices}
        self._v2 = getattr(pynvml, "nvmlMemory_v2", None)
        self.indices = list(indices)

    def name(self, i: int) -> str:
        n = self._n.nvmlDeviceGetName(self._h[i])
        return n.decode() if isinstance(n, bytes) else str(n)

    def mem(self, i: int) -> Tuple[int, int, int]:
        h = self._h[i]
        if self._v2 is not None:
            try:
                m = self._n.nvmlDeviceGetMemoryInfo(h, version=self._v2)
                return int(m.used), int(m.free), int(m.total)
            except Exception:  # noqa: BLE001 -- aeltere Treiber: v1
                self._v2 = None
        m = self._n.nvmlDeviceGetMemoryInfo(h)
        return int(m.used), int(m.free), int(m.total)

    def procs(self, i: int) -> List[Tuple[int, Optional[int]]]:
        out = []
        for p in self._n.nvmlDeviceGetComputeRunningProcesses(self._h[i]):
            used = getattr(p, "usedGpuMemory", None)
            out.append((int(p.pid), None if used is None else int(used)))
        return out

    def close(self) -> None:
        try:
            self._n.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# Rollen-Zuordnung pid -> P:PP0 / D:TP1 / ?:<comm>
# --------------------------------------------------------------------------

#: Rename transition: the scheduler title of either package generation (old and new name
#: written split, so the mechanical rename leaves both in place; RENAME_PLAN 8.15).
_RX_SCHED = re.compile(r"(?:%s|%s)::scheduler_(PP|TP)(\d+)" % ("sg" "lang", "fl" "liper"))
_RX_TAG = re.compile(r"--tag (\S+)")


class RoleResolver:
    """Ordnet eine NVML-pid einer Rolle zu, einmal je pid.

    Reihenfolge: (1) der Prozesstitel ``sglang::scheduler_PP<i>`` / ``_TP<i>``
    nennt den Rang; (2) die Gruppe kommt aus der Ahnenkette gegen
    ``boot_<tag>.json`` ``pids`` (P/D-Gruppenfuehrer), sonst aus dem Titel
    (PP = P, TP = D -- die Form dieses Rigs: P ist PP3, D ist TP3); (3) der Tag
    aus ``--tag`` eines Vorfahren (launcher). Eine pid, die in diesem
    Namensraum nicht existiert, heisst ``?:nopid``."""

    def __init__(self, proc_root: str = "/proc", boot_json: Optional[str] = None,
                 boot_json_dir: str = BOOT_JSON_DIR, tag: Optional[str] = None):
        self.proc_root = proc_root
        self.boot_json = boot_json
        self.boot_json_dir = boot_json_dir
        self.tag = tag
        self.cache: Dict[int, Tuple[str, str]] = {}

    def _read(self, pid: int, name: str) -> str:
        try:
            with open(os.path.join(self.proc_root, str(pid), name), "rb") as f:
                return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            return ""

    def _ppid(self, pid: int) -> int:
        for ln in self._read(pid, "status").splitlines():
            if ln.startswith("PPid:"):
                try:
                    return int(ln.split()[1])
                except (IndexError, ValueError):
                    return 0
        return 0

    def _group_pids(self, tag: Optional[str]) -> Dict[int, str]:
        path = self.boot_json
        if path is None and tag:
            path = os.path.join(self.boot_json_dir, f"boot_{tag}.json")
        if not path:
            return {}
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return {}
        return {int(v): str(k) for k, v in (d.get("pids") or {}).items() if v}

    def role(self, pid: int) -> Tuple[str, str]:
        """(rolle, tag) -- gecacht; eine neue pid kostet einmal /proc-Lesen."""
        hit = self.cache.get(pid)
        if hit is not None:
            return hit
        cmd = self._read(pid, "cmdline")
        if not cmd and not os.path.exists(os.path.join(self.proc_root, str(pid))):
            res = ("?:nopid", self.tag or "")
            self.cache[pid] = res
            return res
        chain = []
        q = pid
        for _ in range(8):
            q = self._ppid(q)
            if q <= 1:
                break
            chain.append(q)
        tag = self.tag
        for a in chain:
            m = _RX_TAG.search(self._read(a, "cmdline"))
            if m:
                tag = m.group(1)
                break
        groups = self._group_pids(tag)
        group = next((groups[a] for a in [pid] + chain if a in groups), None)
        m = _RX_SCHED.search(cmd)
        if m:
            kind, idx = m.group(1), m.group(2)
            if group is None:
                group = "P" if kind == "PP" else "D"
            res = (f"{group}:{kind}{idx}", tag or "")
        else:
            comm = self._read(pid, "comm") or (cmd.split()[0] if cmd else "?")
            comm = re.sub(r"[\s,]+", "_", os.path.basename(comm))[:24]
            res = (f"{group or '?'}:{comm}", tag or "")
        self.cache[pid] = res
        return res


def owner_alive(pattern: str, proc_root: str = "/proc", self_pid: Optional[int] = None) -> bool:
    """Lebt ein Prozess, dessen argv ``pattern`` trifft? (reines /proc-Lesen,
    kein pgrep; die eigene pid und die des Elternprozesses zaehlen nie, weil
    beider argv das Muster als Text tragen koennen)."""
    rx = re.compile(pattern)
    me = {self_pid if self_pid is not None else os.getpid(), os.getppid()}
    try:
        names = os.listdir(proc_root)
    except OSError:
        return False
    for d in names:
        if not d.isdigit() or int(d) in me:
            continue
        try:
            with open(os.path.join(proc_root, d, "cmdline"), "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if rx.search(cmd):
            return True
    return False


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


class BucketAgg:
    """1-s-Eimer je Reihe, an der Unix-Sekunde ausgerichtet. ``add`` fuettert
    JEDE Abtastung (nicht nur Wechsel); beim Sekundenwechsel gibt ``roll``
    die fertigen Zeilen zurueck."""

    HEADER = "utc,sec_unix,key,role,min_mib,max_mib,last_mib,free_min_mib,n"

    def __init__(self, free_of: Callable[[str, int], Optional[int]]):
        self.free_of = free_of
        self.sec: Optional[int] = None
        self.acc: Dict[str, List[int]] = {}  # key -> [min, max, last, n]
        self.roles: Dict[str, str] = {}

    def add(self, unix_ms: int, key: str, v: int, role: str = "card") -> List[str]:
        sec = unix_ms // 1000
        rows: List[str] = []
        if self.sec is not None and sec != self.sec:
            rows = self.flush()
        self.sec = sec
        a = self.acc.get(key)
        if a is None:
            self.acc[key] = [v, v, v, 1]
            self.roles[key] = role
        else:
            if v < a[0]:
                a[0] = v
            if v > a[1]:
                a[1] = v
            a[2] = v
            a[3] += 1
        return rows

    def flush(self) -> List[str]:
        if self.sec is None or not self.acc:
            self.acc = {}
            return []
        utc = time.strftime("%H:%M:%S", time.gmtime(self.sec))
        rows = []
        for key in sorted(self.acc, key=_key_order):
            mn, mx, last, n = self.acc[key]
            free = self.free_of(key, mx)
            rows.append(f"{utc},{self.sec},{key},{self.roles.get(key, '')},{mn},{mx},{last},"
                        f"{'' if free is None else free},{n}")
        self.acc = {}
        return rows


def _key_order(key: str):
    c, _, rest = key.partition(":")
    return (int(c[1:]) if c[1:].isdigit() else 99, rest != "", rest == "other", rest)


class ChangeWriter:
    """Rohstrom verlustfrei fuer Extremwerte: eine Zeile je Wertwechsel einer
    Reihe, dazu ein Herzschlag je ``KEEPALIVE_MS``."""

    def __init__(self, keepalive_ms: int = KEEPALIVE_MS):
        self.keepalive_ms = keepalive_ms
        self.last: Dict[str, Tuple[int, int]] = {}  # key -> (value, t_ms written)

    def feed(self, t_ms: int, key: str, v: int) -> Optional[str]:
        prev = self.last.get(key)
        if prev is not None and prev[0] == v and t_ms - prev[1] < self.keepalive_ms:
            return None
        self.last[key] = (v, t_ms)
        return f"{t_ms},{key},{v}"


class CallStats:
    """Aufrufdauern in us: Mittel, p99 (aus einem Histogramm in 10-us-Stufen bis
    100 ms), Maximum. Konstanter Speicher, egal wie lange die Sonde laeuft."""

    STEP_US = 10
    NBINS = 10_000

    def __init__(self):
        self.n = 0
        self.sum_us = 0.0
        self.max_us = 0.0
        self.bins = [0] * self.NBINS

    def add(self, us: float) -> None:
        self.n += 1
        self.sum_us += us
        if us > self.max_us:
            self.max_us = us
        b = int(us // self.STEP_US)
        self.bins[b if b < self.NBINS else self.NBINS - 1] += 1

    @property
    def mean_us(self) -> float:
        return self.sum_us / self.n if self.n else 0.0

    def pct_us(self, q: float) -> float:
        if not self.n:
            return 0.0
        want = q * self.n
        run = 0
        for i, c in enumerate(self.bins):
            run += c
            if run >= want:
                return float((i + 1) * self.STEP_US)
        return float(self.NBINS * self.STEP_US)


# --------------------------------------------------------------------------
# Die Sonde
# --------------------------------------------------------------------------


class Clock:
    """Monotone ns + Wanduhr + Schlaf; in Tests ersetzt."""

    def mono_ns(self) -> int:
        return time.monotonic_ns()

    def wall_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def sleep(self, s: float) -> None:
        if s > 0:
            time.sleep(s)


class Probe:
    def __init__(self, src, cards: Sequence[int], out_csv: str, raw_path: Optional[str],
                 period_ms: float = DEFAULT_PERIOD_MS, proc_every_ms: float = DEFAULT_PROC_EVERY_MS,
                 budget: float = DEFAULT_BUDGET, resolver: Optional[RoleResolver] = None,
                 clock: Optional[Clock] = None, calib_ticks: int = CALIB_TICKS,
                 argv: Optional[Sequence[str]] = None):
        self.src = src
        self.cards = list(cards)
        self.out_csv = out_csv
        self.raw_path = raw_path
        self.period_ms_req = float(period_ms)
        self.period_ms = float(period_ms)
        self.proc_every_ms = float(proc_every_ms)
        self.budget = float(budget)
        self.resolver = resolver or RoleResolver()
        self.clock = clock or Clock()
        self.calib_ticks = int(calib_ticks)
        self.argv = list(argv or [])
        self.total_mib: Dict[int, int] = {}
        self.offset_mib: Dict[int, int] = {}
        self.names: Dict[int, str] = {}
        self.call_stats = CallStats()      # je MemoryInfo-Aufruf
        self.proc_stats = CallStats()      # je ComputeRunningProcesses-Aufruf
        self.tick_stats = CallStats()      # je Tick (alle Karten)
        self.ticks = 0
        self.proc_ticks = 0
        self.late = 0
        self.max_gap_ms = 0.0
        self.known_pids: Dict[Tuple[int, int], str] = {}
        self.writer = ChangeWriter()
        self.agg = BucketAgg(self._free_of)
        self._stop = False
        self._raw = None
        self._csv = None
        self.t0_ns = 0
        self.t0_unix_ms = 0
        self.stop_reason = ""
        self.header_lines: List[str] = []

    # -- Freiheit einer Karte aus used
    def _free_of(self, key: str, used_max: int) -> Optional[int]:
        if ":" in key or not key.startswith("c"):
            return None
        try:
            i = int(key[1:])
        except ValueError:
            return None
        if i not in self.total_mib:
            return None
        return self.total_mib[i] - self.offset_mib[i] - used_max

    def stop(self, reason: str = "signal") -> None:
        self._stop = True
        self.stop_reason = self.stop_reason or reason

    def _t_ms(self, ns: int) -> int:
        return (ns - self.t0_ns) // 1_000_000

    def _emit_raw(self, line: str) -> None:
        if self._raw is not None:
            self._raw.write(line + "\n")

    def _emit_both(self, line: str) -> None:
        self._emit_raw(line)
        if self._csv is not None:
            self._csv.write(line + "\n")

    # -- ein Tick: alle Karten, optional die Prozesse
    def tick(self, with_procs: bool) -> None:
        t_start = self.clock.mono_ns()
        t_ms = self._t_ms(t_start)
        unix_ms = self.t0_unix_ms + t_ms
        card_used: Dict[int, int] = {}
        for i in self.cards:
            a = self.clock.mono_ns()
            used, _free, _total = self.src.mem(i)
            self.call_stats.add((self.clock.mono_ns() - a) / 1000.0)
            u = used // MIB
            card_used[i] = u
            self._sample(t_ms, unix_ms, f"c{i}", u, "card")
        if with_procs:
            self.proc_ticks += 1
            for i in self.cards:
                a = self.clock.mono_ns()
                try:
                    procs = self.src.procs(i)
                except Exception:  # noqa: BLE001 -- ein Prozess-Lesefehler kostet nur die Reihe
                    procs = []
                self.proc_stats.add((self.clock.mono_ns() - a) / 1000.0)
                attributed = 0
                for pid, used in procs:
                    if used is None:
                        continue
                    u = used // MIB
                    attributed += u
                    role = self._role(t_ms, i, pid)
                    self._sample(t_ms, unix_ms, f"c{i}:{pid}", u, role)
                self._sample(t_ms, unix_ms, f"c{i}:other", card_used[i] - attributed, "other")
        self.tick_stats.add((self.clock.mono_ns() - t_start) / 1000.0)
        self.ticks += 1

    def _sample(self, t_ms: int, unix_ms: int, key: str, v: int, role: str) -> None:
        line = self.writer.feed(t_ms, key, v)
        if line is not None:
            self._emit_raw(line)
        rows = self.agg.add(unix_ms, key, v, role)
        if rows:
            if self._csv is not None:
                self._csv.write("\n".join(rows) + "\n")
            # Eimerwechsel: beide Dateien auf die Platte (SIGKILL verliert <= 1 s)
            self._flush_files()

    def _role(self, t_ms: int, card: int, pid: int) -> str:
        k = (card, pid)
        role = self.known_pids.get(k)
        if role is None:
            role, tag = self.resolver.role(pid)
            self.known_pids[k] = role
            self._emit_both(f"# proc t_ms={t_ms} key=c{card}:{pid} pid={pid} card={card} role={role} tag={tag}")
        return role

    def _flush_files(self) -> None:
        for f in (self._raw, self._csv):
            if f is not None:
                try:
                    f.flush()
                except OSError:
                    pass

    # -- Kalibrierung: Offsets, Aufrufpreis, ggf. Periode anheben
    def calibrate(self) -> None:
        for i in self.cards:
            used, free, total = self.src.mem(i)
            self.total_mib[i] = total // MIB
            self.offset_mib[i] = max(0, (total - used - free) // MIB)
            try:
                self.names[i] = self.src.name(i)
            except Exception:  # noqa: BLE001
                self.names[i] = "?"
        calib = CallStats()
        pcalib = CallStats()
        for _ in range(max(1, self.calib_ticks)):
            for i in self.cards:
                a = self.clock.mono_ns()
                self.src.mem(i)
                calib.add((self.clock.mono_ns() - a) / 1000.0)
        for i in self.cards:
            a = self.clock.mono_ns()
            try:
                self.src.procs(i)
            except Exception:  # noqa: BLE001
                pass
            pcalib.add((self.clock.mono_ns() - a) / 1000.0)
        self.calib = calib
        self.pcalib = pcalib
        tick_cost_ms = (calib.pct_us(0.5) * len(self.cards)
                        + pcalib.mean_us * self.period_ms / max(self.proc_every_ms, self.period_ms)) / 1000.0
        need = tick_cost_ms / self.budget if self.budget > 0 else 0.0
        if need > self.period_ms:
            self.period_ms = float(int(need) + 1)

    def header(self) -> List[str]:
        utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.t0_unix_ms / 1000.0))
        rate = 1000.0 / self.period_ms if self.period_ms > 0 else 0.0
        lines = [
            f"# vram_hires v{FORMAT_VERSION} (fnFL2 H55) t_ms = monotone ms seit t0; unix_ms = t0_unix_ms + t_ms",
            f"# t0_unix_ms={self.t0_unix_ms} t0_utc={utc}Z pid={os.getpid()} argv={' '.join(self.argv)}",
        ]
        for i in self.cards:
            lines.append(f"# card idx={i} name={self.names.get(i, '?').replace(' ', '_')} total_mib={self.total_mib[i]} "
                         f"offset_mib={self.offset_mib[i]} (free = total - offset - used)")
        raised = (f" period_raised_from={self.period_ms_req:g}" if self.period_ms != self.period_ms_req else "")
        lines.append(
            f"# calib rate_hz={rate:.0f} period_ms={self.period_ms:g}{raised} proc_every_ms={self.proc_every_ms:g} "
            f"cards={len(self.cards)} mean_call_us={self.calib.mean_us:.1f} p99_call_us={self.calib.pct_us(0.99):.0f} "
            f"max_call_us={self.calib.max_us:.0f} proc_call_us={self.pcalib.mean_us:.1f} "
            f"tick_budget={self.budget:g} (je MemoryInfo-Aufruf, {self.calib.n} Aufrufe; kein CUDA-Kontext)")
        return lines

    def stat_line(self, t_ms: int, kind: str = "stat") -> str:
        el_s = max(t_ms / 1000.0, 1e-9)
        return (f"# {kind} t_ms={t_ms} ticks={self.ticks} rate_hz={self.ticks / el_s:.1f} "
                f"period_ms={self.period_ms:g} mean_call_us={self.call_stats.mean_us:.1f} "
                f"p99_call_us={self.call_stats.pct_us(0.99):.0f} max_call_us={self.call_stats.max_us:.0f} "
                f"mean_tick_us={self.tick_stats.mean_us:.1f} p99_tick_us={self.tick_stats.pct_us(0.99):.0f} "
                f"proc_ticks={self.proc_ticks} proc_call_us={self.proc_stats.mean_us:.1f} "
                f"busy_pct={100.0 * self.tick_stats.sum_us / 1000.0 / max(t_ms, 1):.2f} "
                f"late={self.late} max_gap_ms={self.max_gap_ms:.1f}"
                + (f" reason={self.stop_reason}" if kind == "end" else ""))

    def open(self) -> None:
        self._csv = open(self.out_csv, "a", buffering=1 << 16)
        if self.raw_path:
            self._raw = open(self.raw_path, "a", buffering=1 << 16)

    def run(self, until: Optional[Callable[[int], Optional[str]]] = None,
            max_ticks: Optional[int] = None) -> None:
        """Hauptschleife. ``until(t_ms)`` -> Grund zum Aufhoeren oder None
        (Eigentuemer-Wache, Hoechstdauer), hoechstens einmal je Sekunde gefragt."""
        self.t0_ns = self.clock.mono_ns()
        self.t0_unix_ms = self.clock.wall_ms()
        self.calibrate()
        self.open()
        self.header_lines = self.header()
        for ln in self.header_lines:
            self._emit_both(ln)
        self._emit_raw("t_ms,key,used_mib")
        if self._csv is not None:
            self._csv.write(BucketAgg.HEADER + "\n")
        self._flush_files()
        period_ns = int(self.period_ms * 1e6)
        proc_every = max(1, int(round(self.proc_every_ms / self.period_ms)))
        next_ns = self.clock.mono_ns()
        last_check = last_stat = last_sync = 0
        prev_tick_ns = None
        k = 0
        while not self._stop:
            now = self.clock.mono_ns()
            if now < next_ns:
                self.clock.sleep((next_ns - now) / 1e9)
                now = self.clock.mono_ns()
            if prev_tick_ns is not None:
                gap_ms = (now - prev_tick_ns) / 1e6
                if gap_ms > self.max_gap_ms:
                    self.max_gap_ms = gap_ms
                if gap_ms > GAP_PERIODS * self.period_ms:
                    self.late += 1
                    self._emit_raw(f"# gap t_ms={self._t_ms(now)} gap_ms={gap_ms:.1f}")
            prev_tick_ns = now
            self.tick(with_procs=(k % proc_every == 0))
            k += 1
            t_ms = self._t_ms(self.clock.mono_ns())
            if t_ms - last_stat >= STAT_EVERY_MS:
                last_stat = t_ms
                self._emit_both(self.stat_line(t_ms))
            if t_ms - last_sync >= SYNC_EVERY_MS:
                last_sync = t_ms
                self._emit_raw(f"# sync t_ms={t_ms} unix_ms={self.clock.wall_ms()}")
            if until is not None and t_ms - last_check >= 1000:
                last_check = t_ms
                why = until(t_ms)
                if why:
                    self.stop(why)
            if max_ticks is not None and self.ticks >= max_ticks:
                self.stop("max_ticks")
            # naechste Deadline im festen Raster; wer mehr als eine Periode
            # hinterher ist, springt vor (kein Aufholen im Burst)
            next_ns += period_ns
            behind = self.clock.mono_ns() - next_ns
            if behind > period_ns:
                next_ns += (behind // period_ns) * period_ns
        self.close()

    def close(self) -> None:
        t_ms = self._t_ms(self.clock.mono_ns())
        if self._csv is not None:
            for row in self.agg.flush():
                self._csv.write(row + "\n")
        self._emit_both(self.stat_line(t_ms, "end"))
        for f in (self._raw, self._csv):
            if f is not None:
                try:
                    f.close()
                except OSError:
                    pass
        self._raw = self._csv = None


# --------------------------------------------------------------------------
# Lebensdauer
# --------------------------------------------------------------------------


def set_pdeathsig(sig: int = signal.SIGTERM) -> bool:
    """PR_SET_PDEATHSIG: stirbt der Elternprozess (der Logger-Shell des Arms),
    bekommt die Sonde ``sig`` -- auch nach einem kill -9 des Arms."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(1, int(sig), 0, 0, 0) == 0  # PR_SET_PDEATHSIG = 1
    except Exception:  # noqa: BLE001
        return False


class OwnerWatch:
    """``until``-Funktion fuer ``Probe.run``: Ende, wenn der Eigentuemer (argv-
    Regex) nie erschien (``grace_s``) oder seit ``gone_s`` fehlt, oder nach
    ``max_s``. Abfrage hoechstens alle ``poll_s``. ``gone_s`` (60) ist lang
    genug fuer die Luecke zwischen Dry-Run-Launcher und Boot-Launcher desselben
    Tags, und kurz gegen die H52-Waisen (die liefen Stunden)."""

    def __init__(self, pattern: Optional[str], poll_s: float = 5.0, grace_s: float = 900.0,
                 max_s: float = 4 * 3600.0, gone_s: float = 60.0, proc_root: str = "/proc",
                 alive: Optional[Callable[[], bool]] = None, parent_pid: Optional[int] = None):
        self.pattern = pattern
        self.poll_ms = int(poll_s * 1000)
        self.grace_ms = int(grace_s * 1000)
        self.max_ms = int(max_s * 1000)
        self.gone_ms = int(gone_s * 1000)
        self.proc_root = proc_root
        self._alive = alive
        self.parent_pid = parent_pid
        self.seen = False
        self.last_seen = 0
        self.last_poll = -10**12

    def alive(self) -> bool:
        if self._alive is not None:
            return self._alive()
        return owner_alive(self.pattern, self.proc_root)

    def __call__(self, t_ms: int) -> Optional[str]:
        if self.max_ms and t_ms >= self.max_ms:
            return "max_s"
        if self.parent_pid is not None and os.getppid() != self.parent_pid:
            return "parent_gone"
        if not self.pattern and self._alive is None:
            return None
        if t_ms - self.last_poll < self.poll_ms:
            return None
        self.last_poll = t_ms
        if self.alive():
            self.seen = True
            self.last_seen = t_ms
            return None
        if not self.seen:
            return "owner_never_seen" if t_ms >= self.grace_ms else None
        return "owner_gone" if t_ms - self.last_seen >= self.gone_ms else None


def raw_path_for(out_csv: str) -> str:
    base = out_csv[:-4] if out_csv.endswith(".csv") else out_csv
    return base + ".raw.csv"


def default_owner_pattern(tag: str) -> str:
    """Wie launcher.memts_owner_pattern (H52): launcher oder front DIESES Tags."""
    return rf"sglang\.srt\.weg2\.(launcher|front) .*--tag {re.escape(tag)}( |$)"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="Eimer-CSV; Rohstrom daneben als .raw.csv")
    ap.add_argument("--cards", default="0,1,2", help="NVML-Indizes")
    ap.add_argument("--period-ms", type=float, default=DEFAULT_PERIOD_MS)
    ap.add_argument("--proc-every-ms", type=float, default=DEFAULT_PROC_EVERY_MS)
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                    help="Tick-Preis hoechstens dieser Anteil der Periode, sonst Periode anheben")
    ap.add_argument("--no-raw", action="store_true", help="nur 1-s-Eimer (Rohstrom ist Standard)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--boot-json", default=None, help="Standard: /spinning/gpu-arb/weg2/boot_<tag>.json")
    ap.add_argument("--owner-pattern", default=None,
                    help="argv-Regex des Eigentuemers; 'auto' = launcher/front dieses --tag")
    ap.add_argument("--owner-poll-s", type=float, default=5.0)
    ap.add_argument("--owner-grace-s", type=float, default=900.0)
    ap.add_argument("--owner-gone-s", type=float, default=60.0)
    ap.add_argument("--max-s", type=float, default=4 * 3600.0)
    ap.add_argument("--nice", type=int, default=10)
    ns = ap.parse_args(argv)
    cards = [int(x) for x in ns.cards.split(",") if x.strip()]
    pattern = ns.owner_pattern
    if pattern == "auto":
        pattern = default_owner_pattern(ns.tag) if ns.tag else None
    set_pdeathsig()
    try:
        os.nice(ns.nice)
    except OSError:
        pass
    src = NvmlSource(cards)
    probe = Probe(src, cards, ns.out, None if ns.no_raw else raw_path_for(ns.out),
                  period_ms=ns.period_ms, proc_every_ms=ns.proc_every_ms, budget=ns.budget,
                  resolver=RoleResolver(boot_json=ns.boot_json, tag=ns.tag),
                  argv=sys.argv if argv is None else list(argv))
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, lambda signum, _f: probe.stop(signal.Signals(signum).name))
    watch = OwnerWatch(pattern, poll_s=ns.owner_poll_s, grace_s=ns.owner_grace_s, max_s=ns.max_s,
                       gone_s=ns.owner_gone_s,
                       parent_pid=os.getppid())
    try:
        probe.run(until=watch)
    finally:
        src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
