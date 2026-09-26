# SPDX-License-Identifier: Apache-2.0
"""#48b -- die Experten-Zeilen je Rang aus der GEMESSENEN Retry-Kante.

DER BEFUND (Register-Audit #48/#81, 26.09.). Das Release-Profil ``nf.env``
traegt ``NF_FR_P=0.332,0.64,0.39`` und ``NF_FR_D=0.06,0.51,0.48`` als
Hand-Pins an drei Stellen. Der Planer rechnet heute je Rang nur DECKEN, die
die Pins nicht erreichen (fnFL2x177, Launcher-Log Z.64-66/195/200, dieselben
Zahlen x176/x178/fnFL2h91v1):

* P-KARTE (H41/H59, W132): 238 / 395 / 576 Zeilen gegen Pin 202 / 360 / 408;
* D-KARTE (H33, W130): 136 / 140 / 141 gegen Pin 130 / 122 / 133;
* WAKE-CREDIT (H14/H34, W126): bindet nur D TP1 (132) und D TP2 (134).

Die Pins sind keine Rate-Werte, sondern die RETRY-KANTE: x167 zog PP0 von
0.36 auf 0.332, weil der Caching-Allokator darueber Anforderungen wiederholt
(x166 PP0 ``alloc_retries`` 1/1/2/4, ~0,6 s je 16k-Chunk; H59b). Die Karten
des Planers messen ``Kopfraum = cap - peak(allocated) - privat_frei`` und
zaehlen damit den Allokator-Cache (``reserved - allocated``, am Metall PP0
~3,6 GiB) als frei -- sie sind die OOM-Kante, nicht die Retry-Kante. Am Pin
misst ``[vram-peak]`` (WEG2-VRAM-PEAK, ``card_free_mib`` = NVML) je Rang:

    x177 P: PP0 min 123 MiB frei, 0 Retries; PP1 84 MiB, 1 Retry; PP2 2682, 0
    x177 D: TP0 285 MiB, 22 Retries; TP1 28, 4; TP2 176, 9 (chunk + round)

DER TERM (keine Reserve, kein Pin -- jede Zahl aus dem Log der Referenz):
je Rang r und Referenz-Boot b mit ``rows_b`` Pufferzeilen (die Zeile
``MoE expert-offload active on layer``) ueber die Fenster der rechnenden
Phasen (P: ``chunk``, D: ``chunk`` und ``round``)

    retries_b == 0:  kante = rows_b + floor(min card_free_b / zeile_r)
    retries_b  > 0:  kante = rows_b - 1   (die Form liegt UEBER der Kante; eine
                     Zeile ist der kleinste Schritt -- der naechste Boot misst
                     nach, bis sie wiederholungsfrei ist)

mit ``zeile_r`` = Layer des Rangs x MiB je Experten-Zeile (die Karte zahlt
jede Zeile ganz). Ueber mehrere Referenz-Boots gilt das Minimum. Die Kante ist
eine Byte-Groesse (limit-neutral, Power-Limit aendert sie nicht) und gilt nur
fuer die FORM der Referenz: Modell, Experten je Rang (Spanne+Pad aus dem Log,
gegen ``--rank-moe-ratio`` des Boots geprueft -- ein Verhaeltnis, keine
Stueckzahl), Layer je Stufe, Sitze (``max_running_requests``) und P-Chunk.
Eine fremde Form ENTFAELLT mit Namen, nie eine halbe Zahl.

Die Fraction folgt aus den Zeilen ueber die Pufferregel (H8,
``expert_residency.largest_fraction_for_rows``): ``R = rows - Scratch``,
``f`` = groesste 3-stellige Fraction mit ``ceil(f x E) <= R`` -- dieselbe
Rundung wie die Runtime (``resident_slot_count``: ceil), nie ``int``.

Alle anderen Terme bleiben RIEGEL auf der aufgeloesten Zahl (#140, W132,
W122, W130, W126): der Launcher setzt die Fractions VOR jedem Leser, die
Riegel pruefen sie wie jede gegebene Zahl.

Rein (kein torch): der Launcher rechnet es ohne CUDA.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec

from sglang.srt.planner import expert_residency as _er

MARKER = "EXPERT-ROWS-AUTO"
#: auto verlangt, aber die Referenz traegt die Form nicht (fehlende Zeilen,
#: andere Geometrie, andere Sitze): keine Zahl, sondern Verweigerung.
REFUSAL_CODE = "W160 Weg2ExpertRowsAutoUnmeasured"
#: Die Phasen, in denen eine Gruppe RECHNET (Flip/Idle priced W126 bzw. nichts).
P_PHASES = ("chunk",)
D_PHASES = ("chunk", "round")

_RX_PEAK = re.compile(
    r"\[[^\]]*?(PP|TP)(\d+)\] WEG2-VRAM-PEAK rank=\d+ phase=(\S+) rows=(\d+) .*?"
    r"card_free_mib=(-?\d+) .*?alloc_retries=(\S+)"
)
_RX_ROWS = re.compile(
    r"\[[^\]]*?(PP|TP)(\d+)\] MoE expert-offload active on layer (\d+): "
    r"(\d+)/(\d+) experts resident \+ (\d+) scratch \(buffer=(\d+), fraction=([0-9.]+)\)"
)
_RX_SEATS = re.compile(r"\[[^\]]*?(PP|TP)(\d+)\] max_total_num_tokens=\d+.*?max_running_requests=(\d+)")


class RankForm(msgspec.Struct, frozen=True, kw_only=True):
    """Was ein Referenz-Log ueber einen Rang selbst sagt."""

    rank: int
    local_experts: int
    resident: int
    scratch: int
    buffer_rows: int
    fraction: float
    layers: int


class RankEdge(msgspec.Struct, frozen=True, kw_only=True):
    rank: int
    source: str
    rows_ref: int
    layers: int
    row_card_mib: float
    windows: int
    free_min_mib: Optional[float]
    retries: int
    edge_rows: Optional[int]
    why: str = ""


def rank_forms(text: str, tag: str) -> Dict[int, RankForm]:
    """``{Rang: RankForm}`` aus den Zeilen ``MoE expert-offload active on
    layer N`` der Gruppe (``tag`` = ``PP`` fuer P, ``TP`` fuer D). Ein Rang,
    dessen Layer verschiedene Puffer melden, ist keine Form -> ValueError."""
    seen: Dict[int, Dict[int, Tuple[int, int, int, int, float]]] = {}
    for m in _RX_ROWS.finditer(text):
        if m.group(1) != tag:
            continue
        r, layer = int(m.group(2)), int(m.group(3))
        seen.setdefault(r, {})[layer] = (
            int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7)),
            float(m.group(8)),
        )
    out: Dict[int, RankForm] = {}
    for r, layers in seen.items():
        forms = set(layers.values())
        if len(forms) != 1:
            raise ValueError(
                "%s%d: die Layer melden %d verschiedene Puffer %s -- keine eine Form"
                % (tag, r, len(forms), sorted(forms)[:3])
            )
        res, E, S, buf, f = forms.pop()
        out[r] = RankForm(rank=r, local_experts=E, resident=res, scratch=S,
                          buffer_rows=buf, fraction=f, layers=len(layers))
    return out


def seats_of(text: str, tag: str) -> Optional[int]:
    """Die Sitze der Gruppe (``max_running_requests`` des ersten Rangs)."""
    for m in _RX_SEATS.finditer(text):
        if m.group(1) == tag:
            return int(m.group(3))
    return None


def chunk_rows_of(text: str, tag: str) -> Optional[int]:
    """Die groesste Fensterzeilenzahl der ``chunk``-Fenster (P: der Chunk)."""
    rows = [int(m.group(4)) for m in _RX_PEAK.finditer(text)
            if m.group(1) == tag and m.group(3) == "chunk"]
    return max(rows) if rows else None


def rank_edges(text: str, *, tag: str, phases: Sequence[str], slot_mib: float,
               source: str) -> Dict[int, RankEdge]:
    """Die Retry-Kante je Rang EINES Referenz-Logs (siehe Modul-Doc)."""
    forms = rank_forms(text, tag)
    free: Dict[int, List[float]] = {}
    retries: Dict[int, int] = {}
    windows: Dict[int, int] = {}
    for m in _RX_PEAK.finditer(text):
        if m.group(1) != tag or m.group(3) not in phases:
            continue
        r = int(m.group(2))
        windows[r] = windows.get(r, 0) + 1
        free.setdefault(r, []).append(float(m.group(5)))
        if m.group(6) != "na":
            retries[r] = retries.get(r, 0) + int(m.group(6))
    out: Dict[int, RankEdge] = {}
    for r, form in sorted(forms.items()):
        row = float(form.layers) * float(slot_mib)
        n = windows.get(r, 0)
        if n == 0:
            out[r] = RankEdge(rank=r, source=source, rows_ref=form.buffer_rows,
                              layers=form.layers, row_card_mib=row, windows=0,
                              free_min_mib=None, retries=0, edge_rows=None,
                              why="keine WEG2-VRAM-PEAK-Fenster %s" % list(phases))
            continue
        fmin = min(free[r])
        k = int(retries.get(r, 0))
        edge = (form.buffer_rows - 1 if k > 0
                else form.buffer_rows + int(math.floor(max(0.0, fmin) / row)))
        out[r] = RankEdge(rank=r, source=source, rows_ref=form.buffer_rows,
                          layers=form.layers, row_card_mib=row, windows=n,
                          free_min_mib=fmin, retries=k,
                          edge_rows=min(edge, form.local_experts))
    return out


class GroupAuto(msgspec.Struct, frozen=True, kw_only=True):
    group: str
    rows: Tuple[int, ...]
    fractions: Tuple[float, ...]
    edges: Tuple[Tuple[RankEdge, ...], ...]
    lines: Tuple[str, ...]
    refusal: Optional[str] = None


def _refused(group: str, why: str) -> GroupAuto:
    text = "%s %s: auto verlangt, %s -- keine Zahl ohne gemessene Form" % (
        REFUSAL_CODE, group, why)
    return GroupAuto(group=group, rows=(), fractions=(), edges=(),
                     lines=("%s %s" % (MARKER, text),), refusal=text)


def solve_group(*, group: str, reference_texts: Sequence[Tuple[str, str]], tag: str,
                phases: Sequence[str], slot_mib: float, local_experts: Sequence[int],
                layers: Sequence[int], scratch: Sequence[int], seats: int,
                chunk: Optional[int] = None) -> GroupAuto:
    """Die auto-Zeilen und -Fractions EINER Gruppe gegen die Referenz-Logs.

    ``local_experts``/``layers``/``scratch``/``seats``/``chunk`` sind die Form
    DIESES Boots (E je Rang aus seiner Ratio + Pad, Layer je Stufe, Scratch aus
    seiner Env, Sitze aus seinem argv, P-Chunk). Jede Abweichung der Referenz
    entfaellt mit Namen (``refusal``)."""
    n = len(local_experts)
    if not (len(layers) == len(scratch) == n) or n == 0:
        return _refused(group, "halbe Geometrie: E %s, Layer %s, Scratch %s"
                        % (list(local_experts), list(layers), list(scratch)))
    if not reference_texts:
        return _refused(group, "keine Referenz-Logs")
    per_boot: List[Dict[int, RankEdge]] = []
    for source, text in reference_texts:
        try:
            forms = rank_forms(text, tag)
        except ValueError as exc:
            return _refused(group, "%s: %s" % (source, exc))
        if sorted(forms) != list(range(n)):
            return _refused(group, "%s: Raenge %s im Log, der Boot hat %d"
                            % (source, sorted(forms), n))
        got_e = [forms[r].local_experts for r in range(n)]
        got_l = [forms[r].layers for r in range(n)]
        if got_e != [int(x) for x in local_experts] or got_l != [int(x) for x in layers]:
            return _refused(group, "%s: fremde Geometrie E %s Layer %s, der Boot E %s Layer %s"
                            % (source, got_e, got_l, list(local_experts), list(layers)))
        ref_seats = seats_of(text, tag)
        if ref_seats is None or int(ref_seats) != int(seats):
            return _refused(group, "%s: Sitze %s, der Boot %d" % (source, ref_seats, int(seats)))
        if chunk is not None:
            ref_chunk = chunk_rows_of(text, tag)
            if ref_chunk is None or int(ref_chunk) != int(chunk):
                return _refused(group, "%s: Chunk %s, der Boot %d" % (source, ref_chunk, int(chunk)))
        edges = rank_edges(text, tag=tag, phases=phases, slot_mib=slot_mib, source=source)
        missing = [r for r in range(n) if edges[r].edge_rows is None]
        if missing:
            return _refused(group, "%s: Rang %s %s" % (
                source, missing, "; ".join(edges[r].why for r in missing)))
        per_boot.append(edges)
    rows: List[int] = []
    fracs: List[float] = []
    lines: List[str] = []
    for r in range(n):
        es = [b[r] for b in per_boot]
        bind = min(es, key=lambda e: int(e.edge_rows))
        R_max = int(bind.edge_rows)
        E, S = int(local_experts[r]), int(scratch[r])
        f = _er.largest_fraction_for_rows(local_experts=E, scratch_rows=S, max_rows=R_max)
        if f is None:
            return _refused(group, "Rang %d: Kante %d Zeilen traegt nicht einmal einen "
                            "residenten Experten + Scratch %d" % (r, R_max, S))
        got = _er.buffer_rows(local_experts=E, fraction=f, scratch_rows=S)
        rows.append(got)
        fracs.append(f)
        lines.append(
            "%s %s rang%d: %s | Kante %d Zeilen (%s) -> f %.3f = R %d + Scratch %d = %d "
            "Zeilen (Pufferregel H8, ceil wie die Runtime)" % (
                MARKER, group, r,
                "; ".join(
                    "%s %d Zeilen x %d Layer (%.1f MiB/Zeile auf der Karte), %d Fenster %s, "
                    "min frei %.0f MiB, Retries %d -> %s" % (
                        e.source, e.rows_ref, e.layers, e.row_card_mib, e.windows,
                        "/".join(phases), e.free_min_mib, e.retries,
                        ("UEBER der Kante, %d" % e.edge_rows) if e.retries
                        else ("%d + %d" % (e.rows_ref, int(e.edge_rows) - e.rows_ref)))
                    for e in es),
                R_max, bind.source, f, _er.resident_rows(E, f), S, got))
    head = ("%s %s: Zeilen %s -> Fractions %s (E %s, Scratch %s, Sitze %d%s). Term: "
            "Retry-Kante aus [vram-peak] (NVML card_free + alloc_retries der rechnenden "
            "Fenster), Byte-Groesse, limit-neutral; alle anderen Planer-Terme (#140, W132, "
            "W122, W130, W126) pruefen diese Zahl als Riegel." % (
                MARKER, group, rows, ",".join("%.3f" % x for x in fracs),
                list(local_experts), list(scratch), int(seats),
                (", Chunk %d" % int(chunk)) if chunk is not None else ""))
    return GroupAuto(group=group, rows=tuple(rows), fractions=tuple(fracs),
                     edges=tuple(tuple(b[r] for b in per_boot) for r in range(n)),
                     lines=(head,) + tuple(lines))


def pin_delta_line(group: str, auto: GroupAuto, pinned: Sequence[float],
                   local_experts: Sequence[int], scratch: Sequence[int],
                   row_card_mib: Sequence[float]) -> str:
    """Ein gegebener Vektor ist ein OVERRIDE: die Zeile nennt ihn als Hand-Pin
    (Planer-Schuld #48) und seinen Abstand zu auto in Zeilen und MiB je Rang."""
    rows = [_er.buffer_rows(local_experts=int(E), fraction=float(f), scratch_rows=int(S))
            for E, f, S in zip(local_experts, pinned, scratch)]
    parts = []
    for r, (a, p, m) in enumerate(zip(auto.rows, rows, row_card_mib)):
        parts.append("rang%d %d gegen auto %d (%+d Zeilen, %+.0f MiB)"
                     % (r, p, a, p - a, (p - a) * float(m)))
    return ("%s %s OVERRIDE: gegebener Vektor %s = HAND-PIN (Planer-Schuld #48), er gilt; "
            "%s" % (MARKER, group, ",".join("%g" % float(x) for x in pinned), "; ".join(parts)))


# ---------------------------------------------------------------------------
# Veroeffentlichen: EIN Wert an allen Stellen, an denen die Gruppe ihn liest
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    return ("%.6f" % float(v)).rstrip("0").rstrip(".") if float(v) != 0 else "0"


def is_auto(value: Optional[str]) -> bool:
    """Leer oder ``auto`` = der Planer loest; alles andere ist ein Override."""
    return str(value or "").strip().lower() in ("", "auto")


def set_flag(extra: str, flag: str, vector: Sequence[float], *, shlex_split, shlex_join) -> str:
    """``extra`` mit ``flag vector``: jede vorhandene Kopie ersetzt (argparse nimmt
    die letzte), sonst angehaengt."""
    toks = list(shlex_split(str(extra or "")))
    val = ",".join(_fmt(v) for v in vector)
    out: List[str] = []
    found = False
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == flag and i + 1 < len(toks):
            out.extend([t, val])
            found = True
            i += 2
            continue
        if t.startswith(flag + "="):
            out.append("%s=%s" % (flag, val))
            found = True
        else:
            out.append(t)
        i += 1
    if not found:
        out.extend([flag, val])
    return shlex_join(out)


def set_env(spec: str, key: str, vector: Sequence[float]) -> str:
    """``'K=V;K=V'`` mit ``key=vector`` (ersetzt, sonst angehaengt)."""
    items = [x for x in str(spec or "").split(";") if x.strip()]
    val = ",".join(_fmt(v) for v in vector)
    out = []
    found = False
    for item in items:
        k, _, _v = item.partition("=")
        if k.strip() == key:
            out.append("%s=%s" % (key, val))
            found = True
        else:
            out.append(item)
    if not found:
        out.append("%s=%s" % (key, val))
    return ";".join(out)


def flag_value(extra: str, flag: str, *, shlex_split) -> Optional[str]:
    """Der LETZTE Wert von ``flag`` in ``extra`` (argparse), sonst None."""
    toks = list(shlex_split(str(extra or "")))
    val = None
    for i, t in enumerate(toks):
        if t == flag and i + 1 < len(toks):
            val = toks[i + 1]
        elif t.startswith(flag + "="):
            val = t.split("=", 1)[1]
    return val


def env_value(spec: str, key: str) -> Optional[str]:
    val = None
    for item in str(spec or "").split(";"):
        k, _, v = item.partition("=")
        if k.strip() == key:
            val = v.strip()
    return val


def fractions_of(text: Optional[str]) -> List[float]:
    return [float(x) for x in str(text or "").split(",") if x.strip()]


def ratio_spans(num_experts: int, ratios: Sequence[float], pad: int = 1) -> List[int]:
    """E je D-Rang: Spanne aus dem VERHAELTNIS (Largest-Remainder, Summe ==
    num_experts geprueft) + Pad-Zeile."""
    spans = _er.expert_span_by_rank(num_experts=int(num_experts), ratios=list(ratios))
    if sum(spans) != int(num_experts):
        raise ValueError("Spannen %s summieren %d, nicht %d" % (spans, sum(spans), num_experts))
    return [int(s) + int(pad) for s in spans]
