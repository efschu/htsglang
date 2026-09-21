"""One shared host store per (layer, expert tensor), rows = GLOBAL expert ids
(Task #47 build step 1, 19.09.).

Today every rank pins a PRIVATE spill pool per layer and tensor whose row j
is ``plan.spill_ids[j]`` (a LOCAL id of the uneven expert-dim shard). Two
rank groups that must both reach the whole expert set -- the TP3 decode group
(disjoint expert ranges per rank) and the PP3 prefill group (every expert of
its layers) -- would need two host copies, 98 GB against the 88 GB mark. With
the store there is ONE tmpfs file per (layer, tensor) of shape
``(num_experts, *row)``: every rank writes the rows it loaded (residents and
spill alike) at ``lo + local - 1``, overlapping writers write identical
bytes, and a reader from any group indexes the same file by global id. The
device pool's row copy (``expert_pool_device.copy_rows``) needs exactly one
contiguous host source per tensor, which is why this is one file per tensor
and not the cold tier's one segment per owner (#394 slice 2).

Off unless ``SGLANG_MOE_EXPERT_STORE_DIR`` names a directory. Pure file and
tensor arithmetic here; ``shared_pinned`` does the mmap + cudaHostRegister.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.moe.shared_pinned import shared_pinned_empty

STORE_DIR_ENV = "SGLANG_MOE_EXPERT_STORE_DIR"

#: #72 (Nutzer-Order 21.09.): WIEVIELE PLAETZE der Store haelt -- als Anteil
#: der Expertenzahl, oder leer fuer "alle" (das bisherige Verhalten).
#:
#: "auf den karten liegt IMMER ein teil des modells, der rest liegt im
#: systemram. aber waehrend decode oder prefill muss niemals alles im
#: systemram liegen."
#:
#: Es kommt auf die ANZAHL an, nicht auf die Identitaet: liegen zu jedem
#: Zeitpunkt R von N Experten auf den Karten, braucht der Store nie mehr als
#: N-R Plaetze -- auch wenn dauernd andere Experten darin stehen. Gemessen an
#: fnFL2w5/w7 haelt er heute alle 512: 59 GiB auf Platte, 61,10 GiB shmem,
#: der groesste nicht-reclaimable Posten gegen die ~93-GiB-Decke, an der w4
#: mit oom_kill starb.
SLOT_FRACTION_ENV = "SGLANG_MOE_EXPERT_STORE_SLOT_FRACTION"

#: #91: DIE EINE GEOMETRIE, die beide Ranggruppen teilen muessen.
#:
#: Der Store ist EINE Datei je Layer/Attribut, geteilt zwischen P und D. Jede
#: Gruppe rechnet ihre Slot-Zuordnung heute aus IHREN eigenen Vektoren --
#: P (tp1/pp3) kommt auf 358 Plaetze, D (183,137,168) auf 332. Verschiedene
#: Zuordnungen auf denselben Bytes sind kein Speicherverlust, sondern
#: DATENVERLUST: D schreibt dann in die Zeilen, aus denen P nach dem Wake
#: liest.
#:
#: Deshalb gibt der Launcher -- der als einziger BEIDE Konfigurationen kennt
#: -- eine gemeinsame Geometrie vor, und beide Gruppen rechnen daraus
#: dieselbe Abbildung `globale Id -> Slot`. Format: "r0,r1,r2|f0,f1,f2".
STORE_GEOMETRY_ENV = "SGLANG_MOE_EXPERT_STORE_GEOMETRY"

__all__ = [
    "STORE_DIR_ENV",
    "store_dir",
    "store_enabled",
    "store_path",
    "open_store",
    "slot_fraction",
    "slots_for",
    "global_rows",
    "slot_rows",
    "slot_base_for_rank",
    "shared_geometry",
    "write_rows",
    "mark_rows_written",
    "rows_written",
    "written_rows_cached",
    "forget_written_rows",
    "store_has_row",
]


def store_dir() -> str:
    return os.environ.get(STORE_DIR_ENV, "").strip()


def store_enabled() -> bool:
    return bool(store_dir())


def slot_fraction() -> float:
    """Anteil der Experten, fuer die der Store Plaetze haelt (0 < f <= 1).

    KONSERVATIV: alles, was nicht als Zahl in (0, 1] lesbar ist -- leer,
    Unsinn, 0, negativ, > 1 -- ergibt 1.0, also das bisherige Verhalten mit
    einem Platz je Experte. Ein Irrtum in diese Richtung kostet Host-RAM; der
    Irrtum in die andere Richtung liesse einen Experten ohne Platz zurueck.
    """
    raw = os.environ.get(SLOT_FRACTION_ENV, "").strip()
    if not raw:
        return 1.0
    try:
        f = float(raw)
    except ValueError:
        return 1.0
    if not (0.0 < f <= 1.0):
        return 1.0
    return f


def slots_for(num_experts: int, fraction: float | None = None) -> int:
    """Wieviele PLAETZE fuer ``num_experts`` Experten -- mindestens einer.

    AUFGERUNDET: lieber ein Platz zuviel als ein Experte ohne Platz.
    """
    f = slot_fraction() if fraction is None else float(fraction)
    n = int(num_experts)
    if n <= 0:
        return 0
    import math

    return max(1, min(n, int(math.ceil(n * f))))


def store_path(directory: str, layer_key: str, attr: str) -> str:
    safe = str(layer_key).replace("/", "_").replace(".", "_")
    return os.path.join(directory, f"{safe}-{attr}.bin")


def open_store(
    directory: str,
    layer_key: str,
    attr: str,
    num_experts: int,
    row_shape: Sequence[int],
    dtype: torch.dtype,
    register=None,
    num_slots: Optional[int] = None,
) -> Tuple[torch.Tensor, bool]:
    """The ``(slots, *row_shape)`` tensor backed by the store file.
    Returns (tensor, created): the first opener creates the file (zero pages),
    every later one maps the same bytes.

    #72: ``slots`` ist per Default ``num_experts`` -- byte-identisch zu
    frueher. Steht ``SLOT_FRACTION_ENV`` (oder ``num_slots``), haelt der
    Store WENIGER Plaetze als es Experten gibt, weil ein Teil der Experten
    immer auf den Karten liegt und dort nie aus dem Host gelesen wird. Die
    Zuordnung Experte -> Platz ist dann nicht mehr die Identitaet; sie gehoert
    dem Aufrufer, der die Verdraengung kennt.
    """
    os.makedirs(directory, exist_ok=True)
    path = store_path(directory, layer_key, attr)
    slots = int(num_slots) if num_slots is not None else slots_for(int(num_experts))
    return shared_pinned_empty(
        path, (int(slots),) + tuple(int(d) for d in row_shape), dtype, register
    )


def global_rows(local_ids: Iterable[int], lo: int, pad: bool = True) -> Dict[int, int]:
    """Local id -> store row. ``pad=True`` is the generic expert-dim shard:
    local 0 is the zero pad expert (no row), local i >= 1 is global
    ``lo + i - 1``. ``pad=False`` is an unsharded layer (a PP stage holding
    every expert): local i is global ``lo + i``."""
    out: Dict[int, int] = {}
    for e in local_ids:
        e = int(e)
        if pad:
            if e >= 1:
                out[e] = int(lo) + e - 1
        else:
            out[e] = int(lo) + e
    return out


RESIDENT_IDS_ENV = "SGLANG_MOE_EXPERT_STORE_RESIDENT_IDS"


def shared_resident_ids():
    """Die GLOBALEN Experten-Ids, die BEIDE Gruppen resident halten, oder ``None``.

    #95, gemessen fnFL2w24: P (tp1) schrieb 324 Plaetze, D (tp3) wollte 512
    -- und jeder D-Rang rechnete eine eigene Zahl (421/467/463). Der Grund
    ist eine Annahme in #91/3: dort galt das Hotset EINES Rangs als die
    ganze residente Menge. Unter tp1 stimmt das, denn ein Rang haelt alle
    512 Experten. Unter tp3 kennt jeder Rang nur seinen Bereich, also
    rechnet er 512 minus SEINE 91 statt 512 minus ALLE 188.

    Die Plaetzezahl des Stores ist aber eine GLOBALE Groesse: eine Datei je
    Layer und Attribut, geteilt ueber Raenge UND Gruppen. Deshalb gibt der
    Launcher sie vor -- er ist die einzige Stelle, die beide Gruppen kennt,
    genau wie bei :func:`shared_geometry`.

    Als Komma-Liste oder als Pfad zu einer JSON-Liste. KONSERVATIV: alles
    Unlesbare, Negative oder Leere gibt ``None``, dann bleibt es beim
    bisherigen Weg. Eine falsche globale Menge waere eine falsche
    Slot-Zuordnung, und die ist Datenverlust, nicht Speicherverlust.
    """
    raw = os.environ.get(RESIDENT_IDS_ENV, "").strip()
    if not raw:
        return None
    try:
        if raw.startswith("[") or os.path.sep in raw:
            if not raw.startswith("["):
                with open(raw) as fh:
                    roh = json.load(fh)
            else:
                roh = json.loads(raw)
        else:
            roh = [x for x in raw.split(",")]
        if not isinstance(roh, (list, tuple)):
            return None
        ids = set()
        for x in roh:
            t = str(x).strip()
            if not t or not t.lstrip("+").isdigit():
                return None
            ids.add(int(t))
    except (OSError, ValueError, TypeError):
        return None
    if not ids or any(i < 0 for i in ids):
        return None
    return frozenset(ids)


def shared_geometry():
    """Die gemeinsame Store-Geometrie ``(ratios, fractions)`` oder ``None``.

    #91. Gesetzt vom Launcher ueber ``STORE_GEOMETRY_ENV``, weil nur er beide
    Ranggruppen kennt. Steht sie, rechnen P und D DIESELBE Abbildung
    ``globale Id -> Slot`` -- unabhaengig davon, wie die eigene Gruppe ihre
    Experten aufteilt. Steht sie nicht, bleibt alles wie bisher.

    KONSERVATIV: unlesbare oder widerspruechliche Angaben geben ``None``
    zurueck, nicht eine halbe Geometrie. Eine falsche gemeinsame Zuordnung
    waere schlimmer als gar keine -- sie sieht aus wie Ersparnis und ist
    Datenverlust.
    """
    raw = os.environ.get(STORE_GEOMETRY_ENV, "").strip()
    if not raw or "|" not in raw:
        return None
    left, _, right = raw.partition("|")
    try:
        ratios = [int(x) for x in left.split(",") if x.strip()]
        fracs = [float(x) for x in right.split(",") if x.strip()]
    except (TypeError, ValueError):
        return None
    if not ratios or len(ratios) != len(fracs):
        return None
    if any(r <= 0 for r in ratios) or any(not (0.0 <= f <= 1.0) for f in fracs):
        return None
    return ratios, fracs


def slot_base_for_rank(ratios: Sequence[int],
                       fractions: Sequence[float],
                       rank: int) -> int:
    """Der erste Slot, der DIESEM Rang gehoert (#72).

    Der Store ist ueber Raenge UND Gruppen geteilt, also muessen zwei Raenge
    fuer dieselbe globale Id denselben Platz errechnen -- ohne miteinander zu
    reden. Die beiden Vektoren, die das erlauben, stehen ohnehin auf jeder
    Kommandozeile: ``--rank-moe-ratio`` (wieviele Experten je Rang, GEMESSEN
    60,226,226 auf diesem Rig) und ``--rank-moe-resident-fraction`` alias
    ``SGLANG_MOE_RESIDENT_EXPERT_FRACTION``, ein EnvFloatVector, also schon
    heute ein Wert JE RANG.

    Kalt je Rang ist ``bereich - round(bereich * fraction)``; der Offset ist
    die Summe der Kalten DAVOR. Damit ist der Slot-Raum genau so gross wie
    das, was nie auf einer Karte liegt -- die Nutzer-Order vom 21.09.

    KONSERVATIV: fehlt ein Eintrag oder ist er unlesbar, gilt fraction 0.0
    fuer diesen Rang, also die volle Bereichsgroesse als kalt. Zu viel Platz
    kostet Host-RAM; zu wenig kollidiert, und Kollision ist Datenverlust.
    """
    base = 0
    for r in range(int(rank)):
        try:
            span = int(ratios[r])
        except (IndexError, TypeError, ValueError):
            continue
        if span <= 0:
            continue
        try:
            f = float(fractions[r])
        except (IndexError, TypeError, ValueError):
            f = 0.0
        if not (0.0 <= f <= 1.0):
            f = 0.0
        base += span - int(round(span * f))
    return int(base)


def slot_rows(local_ids: Iterable[int], lo: int, pad: bool = True,
              *, resident_ids: Optional[Iterable[int]] = None,
              num_experts: Optional[int] = None) -> Dict[int, int]:
    """Local id -> SLOT im Slot-Pool (#72). Deterministisch, ohne Absprache.

    `global_rows` nimmt die globale Experten-Id ALS Zeilenindex -- deshalb
    braucht die Datei heute einen Platz je Experte, auch fuer die, die nie
    geschrieben werden. Nutzer-Order 21.09.: "waehrend decode oder prefill
    muss niemals alles im systemram liegen"; Plaetze braucht nur, wer NICHT
    auf einer Karte liegt.

    DIE ZUORDNUNG BLEIBT EINE RECHNUNG, KEINE ABSPRACHE, und das ist die
    ganze Kunst hier: der Store ist GETEILT -- ein Rang schreibt, ein Rang
    der anderen Gruppe liest. Eine Freiliste muesste ueber Prozessgrenzen
    hinweg konsistent sein (ein Konsens, ein Schloss, eine neue
    Fehlerquelle). Stattdessen ist der Slot die POSITION der globalen Id in
    der aufsteigenden Liste der Nicht-Residenten: jeder Rang rechnet sie aus
    denselben zwei Angaben aus (``num_experts``, ``resident_ids``), die der
    Plan ohnehin repliziert, und kommt zwangslaeufig auf dieselbe Zahl.

    Ohne ``resident_ids`` faellt die Funktion auf `global_rows` zurueck --
    byte-identisch zu heute, damit ein Aufrufer ohne Residenzwissen nichts
    kaputt macht.
    """
    rows = global_rows(local_ids, lo, pad)
    if resident_ids is None or num_experts is None:
        return rows
    res = {int(e) for e in resident_ids}
    cold = [e for e in range(int(num_experts)) if e not in res]
    slot_of = {gid: i for i, gid in enumerate(cold)}
    out: Dict[int, int] = {}
    for local, gid in rows.items():
        slot = slot_of.get(int(gid))
        if slot is None:
            # Ein Resident hat keinen Platz -- und soll keinen bekommen. Der
            # Aufrufer schreibt ihn nicht (spill_ids schliesst Residenten
            # aus, expert_offload.py:1529); kaeme er doch, waere das Weglassen
            # der sichere Ausgang: lieber eine Zeile fehlt im Store (der
            # Leser holt sie von der Karte) als eine fremde ueberschrieben.
            continue
        out[local] = int(slot)
    return out


def write_rows(
    store: torch.Tensor,
    src: torch.Tensor,
    local_ids: Sequence[int],
    lo: int,
    pad: bool = True,
    rows: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """Copy ``src[local]`` into its store row for every local id (``src`` is
    expert-major over the rank's LOCAL ids, on any device). Returns the
    local -> row map that was written.

    ``rows`` ist die FERTIGE Abbildung ``lokal -> Zeile``. Ohne sie faellt die
    Funktion auf :func:`global_rows` zurueck, also Zeile == globale Id.

    #94, gemessen fnFL2w22 und w23: genau dieser Rueckfall war der Defekt.
    Seit dem Slot-Pool (#72) hat die Datei nur noch so viele Plaetze, wie
    Experten gleichzeitig kalt sind (324 statt 512), und
    ``_expert_store_rows_for`` rechnet dafuer die Abbildung
    ``lokal -> Slot``. Sie erreichte den LESER (``layer._moe_offload_store_
    index``) und nicht den SCHREIBER -- der rief ``global_rows`` ein zweites
    Mal und schrieb in Zeile 370 einer 324-Zeilen-Datei. Wer rechnet, gibt
    das Ergebnis weiter; zweimal rechnen heisst irgendwann verschieden
    rechnen.
    """
    rows = dict(rows) if rows is not None else global_rows(local_ids, lo, pad)
    if not rows:
        return rows
    kapazitaet = int(store.shape[0])
    daneben = sorted(r for r in rows.values() if not (0 <= int(r) < kapazitaet))
    if daneben:
        raise RuntimeError(
            f"#94: {len(daneben)} Zeilen liegen ausserhalb der Store-Datei "
            f"(0..{kapazitaet-1}), erste: {daneben[:4]}. Entweder ist die "
            f"Abbildung lokal->Slot nicht durchgereicht worden (dann sind es "
            f"globale Ids), oder die Datei wurde mit einer anderen "
            f"Platzzahl angelegt als die Abbildung annimmt."
        )
    locals_ = list(rows.keys())
    dst = torch.as_tensor([rows[e] for e in locals_], dtype=torch.long)
    # NO device-side gather: fn8m (20.09.) showed the index_select copies of
    # every layer's expert tensors staying RESERVED in the caching allocator
    # (+1.9 GiB on the x4 3080), which the KV sizer then read as used and
    # the first 8k chunk OOMed on. Move the rows through the host instead.
    if src.device.type == "cpu":
        store.index_copy_(0, dst, src[torch.as_tensor(locals_, dtype=torch.long)].to(store.dtype))
        return rows
    # fn8m4 (20.09.): a host intermediate (src.to("cpu")) per tensor left the
    # load's anon footprint at 60 GiB while the store filled (shmem 26 GiB)
    # -> 90 GiB against the 88 mark. Copy each row straight from the device
    # into the registered store mapping (D2H DMA), no host buffer at all.
    assert src.dtype == store.dtype, (src.dtype, store.dtype)
    for e in locals_:
        store[rows[e]].copy_(src[e], non_blocking=False)
    return rows


#: #75: der LESER des Sentinels, den es bisher nicht gab.
#:
#: `mark_rows_written` publiziert seit dem ersten Tag, welche Zeilen ein Rang
#: geschrieben hat -- und `rows_written` hatte NULL AUFRUFER (devindex, 21.09.).
#: Die Folge, gemessen an fnFL2w1: Gruppe D liest den ganzen Checkpoint ein
#: zweites Mal (237 s, ~14 GiB Page-Cache, der die Container-Decke bricht),
#: obwohl P dieselben Bytes im geteilten tmpfs liegen hat.
#:
#: Ein Cache je (dir, layer_key, attr): die Sentinels sind Dateien, und der
#: Ladepfad fragt je Experte einmal -- 512 stat()+json.load() je Tensor waere
#: die Ersparnis wieder aufgefressen. Der Cache lebt fuer die Dauer des Ladens
#: und wird von `forget_written_rows` verworfen, wenn jemand schreibt.
_ROWS_CACHE: Dict[Tuple[str, str, str, int], Dict[int, int]] = {}


def written_rows_cached(directory: str, layer_key: str, attr: str,
                        world: int) -> Dict[int, int]:
    """``rows_written`` mit Cache je (dir, layer_key, attr, world)."""
    key = (str(directory), str(layer_key), str(attr), int(world))
    hit = _ROWS_CACHE.get(key)
    if hit is None:
        hit = rows_written(directory, layer_key, attr, world)
        _ROWS_CACHE[key] = hit
    return hit


def forget_written_rows() -> None:
    """Den Cache verwerfen -- nach jedem Schreiben, damit ein Leser nie eine
    Sentinel-Lage von vor dem Schreiben sieht."""
    _ROWS_CACHE.clear()


def store_has_row(layer_key: str, attr: str, global_row: int,
                  world: int) -> bool:
    """Liegt diese GLOBALE Zeile schon im geteilten Store?

    KONSERVATIV PER KONSTRUKTION: jede Unsicherheit -- Store aus, Verzeichnis
    unlesbar, Sentinel fehlt, Zeile nicht darin -- antwortet False, also
    "lies sie vom Checkpoint". Ein Irrtum in diese Richtung kostet Ladezeit;
    der Irrtum in die andere Richtung laedt ein Modell mit einer Luecke und
    rechnet still falsch.
    """
    if not store_enabled():
        return False
    try:
        rows = written_rows_cached(store_dir(), layer_key, attr, int(world))
    except Exception:  # noqa: BLE001 -- ein unlesbarer Sentinel ist ein Lader,
        return False   # kein Absturz: der Checkpoint ist immer noch da.
    return int(global_row) in rows


def _sentinel(directory: str, layer_key: str, attr: str, rank: int) -> str:
    return store_path(directory, layer_key, attr) + f".r{int(rank)}.written.json"


def mark_rows_written(directory: str, layer_key: str, attr: str, rank: int, rows: Iterable[int]) -> str:
    """Publish which store rows ``rank`` finished writing (atomic rename)."""
    path = _sentinel(directory, layer_key, attr, rank)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"rank": int(rank), "rows": sorted(int(r) for r in rows)}, fh)
    os.replace(tmp, path)
    # #75: der Cache des Lesers darf eine gerade publizierte Zeile nicht
    # verpassen -- er wird hier verworfen, nicht per Zeitstempel geraten.
    forget_written_rows()
    return path


def rows_written(directory: str, layer_key: str, attr: str, world: int) -> Dict[int, int]:
    """Row -> writer rank over every published sentinel of the group; a row
    two ranks both claim keeps the lower rank (identical bytes by contract)."""
    out: Dict[int, int] = {}
    for rank in range(int(world) - 1, -1, -1):
        path = _sentinel(directory, layer_key, attr, rank)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            data = json.load(fh)
        for r in data.get("rows", []):
            out[int(r)] = int(rank)
    return out
