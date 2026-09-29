"""#91 Baustein 2: die Store-Slots als TAUSCH, nicht als Rechnung.

Nutzer-Order 21.09. 21:02Z:

    "die hot experts auf den karten wechseln ja, also muss das ein tausch
     von slot auf karte zu slot in vram sein"

Und genau daran scheitert die heutige Rechnung. ``slot_base_for_rank``
leitet den Platz eines Experten STATISCH aus ``--rank-moe-ratio`` und
``--rank-moe-resident-fraction`` ab. Das ist nur korrekt, solange die
Residenzmenge FEST ist -- sie ist es nicht: der Device-Pool verdraengt nach
LRU, also wechselt laufend, wer auf der Karte liegt und wer nicht.

Die Folge im geteilten Store (eine Datei je Layer/Attribut fuer BEIDE
Ranggruppen): P rechnet fuer Id 200 einen anderen Platz aus als D, und wer
zweitens schreibt, trifft die Zeilen des ersten. Gemessen fnFL2w16: die
Datei blieb bei 512 Slots, weil der Pool bei P gar nicht lief (#90) -- mit
laufendem Pool und ungleicher Zuordnung waere es Datenverlust geworden.

DIE UMKEHRUNG, die das aufloest: der Platz gehoert nicht dem Experten,
sondern wird VERGEBEN. Wird X heiss, gibt X seinen Platz frei; wird Y kalt,
bekommt Y genau diesen Platz. Die Zahl der Plaetze ist dann die Zahl der
gleichzeitig Kalten -- nie mehr, egal wie oft getauscht wird. Das ist die
Zeile, die in ``expert_store`` schon als Absicht steht:

    "liegen zu jedem Zeitpunkt R von N Experten auf den Karten, braucht der
     Store nie mehr als N-R Plaetze -- auch wenn dauernd andere Experten
     darin stehen"

Diese Datei haelt nur die Buchfuehrung. Sie kennt weder Tensoren noch
CUDA: ``assign`` gibt einen Platz, ``release`` nimmt ihn zurueck, ``swap``
macht beides in einem Zug, damit zwischen den Schritten nie ein Zustand
steht, in dem beide Experten denselben Platz zu haben scheinen.
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional


class SlotExhausted(RuntimeError):
    """Kein freier Platz -- und das ist ein FEHLER, keine Warnung.

    Ein Experte ohne Platz waere ein stiller Verlust seiner Zeile. Der
    Aufrufer muss die Zahl der Plaetze erhoehen oder mehr resident halten;
    beides ist eine Entscheidung, die hier niemand fuer ihn treffen darf.
    """


class SlotLedger:
    """Welcher globale Experte gerade in welchem Store-Platz liegt.

    ``capacity`` ist die Zahl der Plaetze in der Datei. Sie folgt aus der
    Zahl der gleichzeitig Kalten, nicht aus der Zahl der Experten.
    """

    __slots__ = ("_capacity", "_by_expert", "_by_slot", "_free")

    def __init__(self, capacity: int, *, occupied: Optional[Dict[int, int]] = None):
        cap = int(capacity)
        if cap <= 0:
            raise ValueError(f"capacity must be positive, not {capacity}")
        self._capacity = cap
        self._by_expert: Dict[int, int] = {}
        self._by_slot: Dict[int, int] = {}
        # Ein MIN-HEAP, nicht eine Liste: die Vergabe nimmt immer den
        # KLEINSTEN freien Platz, damit eine frisch angelegte Datei von vorne
        # gefuellt wird und die hinteren Seiten unberuehrt bleiben -- auf
        # tmpfs kosten sie dann nichts. Mit einer Liste waere `pop()` der
        # zuletzt freigegebene, und `swap` waere nicht von release+assign zu
        # unterscheiden (gemessen: der Mutant ueberlebte).
        self._free: List[int] = list(range(cap))
        heapq.heapify(self._free)
        for expert, slot in (occupied or {}).items():
            self._take(int(expert), int(slot))

    # -- Lesen ---------------------------------------------------------
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def free_count(self) -> int:
        return len(self._free)

    def slot_of(self, expert: int) -> Optional[int]:
        """Der Platz dieses Experten, oder ``None`` wenn er auf der Karte liegt."""
        return self._by_expert.get(int(expert))

    def expert_at(self, slot: int) -> Optional[int]:
        return self._by_slot.get(int(slot))

    def as_map(self) -> Dict[int, int]:
        """Die Belegung als ``{globale Id: Platz}`` -- fuer die Uebergabe
        ueber den Flip. Eine Kopie, damit niemand versehentlich die
        Buchfuehrung von aussen umschreibt."""
        return dict(self._by_expert)

    # -- Schreiben -----------------------------------------------------
    def _take(self, expert: int, slot: int) -> None:
        if not (0 <= slot < self._capacity):
            raise ValueError(f"Slot {slot} lies outside 0..{self._capacity-1}")
        occupied_rows = self._by_slot.get(slot)
        if occupied_rows is not None and occupied_rows != expert:
            raise ValueError(
                f"Slot {slot} already belongs to expert {occupied_rows}, not {expert}"
            )
        if slot in self._free:
            self._free.remove(slot)
            heapq.heapify(self._free)
        self._by_expert[expert] = slot
        self._by_slot[slot] = expert

    def assign(self, expert: int) -> int:
        """Einen Platz fuer einen kalt gewordenen Experten. Idempotent."""
        expert = int(expert)
        present = self._by_expert.get(expert)
        if present is not None:
            return present
        if not self._free:
            raise SlotExhausted(
                f"all {self._capacity} slots occupied, Expert {expert} would get none"
            )
        slot = heapq.heappop(self._free)
        self._by_expert[expert] = slot
        self._by_slot[slot] = expert
        return slot

    def release(self, expert: int) -> Optional[int]:
        """Den Platz eines heiss gewordenen Experten freigeben."""
        expert = int(expert)
        slot = self._by_expert.pop(expert, None)
        if slot is None:
            return None
        self._by_slot.pop(slot, None)
        heapq.heappush(self._free, slot)
        return slot

    def swap(self, hot: int, cold_ids: int) -> int:
        """DER TAUSCH: ``heiss`` wandert auf die Karte, ``kalt`` in dessen Platz.

        In EINEM Zug, nicht als release+assign: dazwischen gaebe es einen
        Augenblick, in dem der Platz frei aussieht und ein dritter Aufrufer
        ihn bekommen koennte. Und er ist billiger als beides einzeln -- der
        Platz wandert direkt weiter, ohne ueber die Freiliste zu gehen.
        """
        hot, cold_ids = int(hot), int(cold_ids)
        if hot == cold_ids:
            raise ValueError("Swap with itself is no swap")
        slot = self._by_expert.pop(hot, None)
        if slot is None:
            # Der Heisse hatte gar keinen Platz (lag schon auf der Karte).
            # Dann ist es kein Tausch, sondern eine normale Vergabe.
            return self.assign(cold_ids)
        self._by_slot.pop(slot, None)
        alt = self._by_expert.pop(cold_ids, None)
        if alt is not None:
            # Der Kalte hatte schon einen -- den gibt er zurueck.
            self._by_slot.pop(alt, None)
            heapq.heappush(self._free, alt)
        self._by_expert[cold_ids] = slot
        self._by_slot[slot] = cold_ids
        return slot

    def assign_many(self, experts: Iterable[int]) -> Dict[int, int]:
        """Mehrere auf einmal, in aufsteigender Id-Reihenfolge.

        Die Reihenfolge ist festgelegt, damit zwei Prozesse, die mit
        derselben leeren Tafel und derselben Menge starten, dieselbe
        Belegung errechnen -- ohne miteinander zu reden.
        """
        return {e: self.assign(e) for e in sorted({int(x) for x in experts})}


def ledger_for_cold_set(cold: Iterable[int], capacity: Optional[int] = None) -> SlotLedger:
    """Eine Tafel fuer eine bekannte Menge kalter Experten.

    Ohne ``capacity`` genau so gross wie die Menge -- das ist die
    Untergrenze aus der Nutzer-Order: Plaetze nur fuer das, was NICHT auf
    einer Karte liegt.
    """
    ids = sorted({int(x) for x in cold})
    board = SlotLedger(capacity if capacity is not None else max(1, len(ids)))
    board.assign_many(ids)
    return board
