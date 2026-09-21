"""#84: nur die Zeilen pinnen, die dieser Rang wirklich belegt.

Nutzer-Order 21.09., woertlich: "DU GIBST DIE UEBERFLUESSIGEN MOE SEITEN IM
SYSTEMRAM NICHT FREI ... DU HAST DEN SYSTEMRAM UM NULL GESENKT".

Er hat recht, und die Messung sagt, warum. fnFL2w11:

    nominal 58,01 GiB   belegt 58,01 GiB   (100,0 %, 0 Nullseiten)

Geschrieben werden nur die KALTEN Zeilen -- `spill_ids` schliesst die
Residenten aus (expert_offload.py:1529). Belegt sind trotzdem ALLE, weil
`cudaHostRegister(ptr, nbytes, MAPPED)` die GANZE Datei pinnt und Pinning
jede Seite einfaultet. Die residenten Experten haben dadurch eine
Host-Kopie, die niemand schreibt und niemand liest -- sie existiert nur,
weil die Registrierung sie anfasst.

Bei P (0.12/0.3/0.3 auf 171/171/170) sind das 123 von 512 Zeilen = 24 %.

Der Unterschied zum #72-Slot-Pool, der am Metall 0 GiB brachte: DIE DATEI
BEHAELT IHRE GROESSE. Die globale Id bleibt der Zeilenindex, beide
Ranggruppen rechnen dieselbe Adresse, und es braucht keine gemeinsame
Residenzmenge. Nur der INHALT schrumpft.
"""

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.moe import expert_store as es
from sglang.srt.layers.moe.shared_pinned import _page_align


def _allocated(path):
    return os.stat(path).st_blocks * 512


def test_unused_rows_stay_unallocated(monkeypatch):
    """Der Kern: die Datei ist gross, der Inhalt klein."""
    d = tempfile.mkdtemp(dir="/dev/shm")
    t, created = es.open_store(d, "L0", "w13", 64, (4096,), torch.uint8,
                               register=False, rows=[0, 1, 2])
    assert created and t.shape == (64, 4096)
    path = es.store_path(d, "L0", "w13")
    assert os.stat(path).st_size == 64 * 4096, "die Datei behaelt ihre Groesse"
    t[0].fill_(1); t[1].fill_(2); t[2].fill_(3)
    del t
    belegt = _allocated(path)
    assert belegt <= 8 * 4096, (
        f"nur die beruehrten Zeilen duerfen Platz kosten, belegt={belegt}")


def test_ranges_merge_so_the_count_stays_small():
    """450 kalte Zeilen duerfen nicht 450 Registrierungen werden -- die
    Residenten sind je Rangbereich die ERSTEN, also bleiben wenige
    zusammenhaengende Spannen."""
    rows = list(range(21, 171)) + list(range(222, 342)) + list(range(393, 512))
    ranges = [(r * 4096, (r + 1) * 4096) for r in rows]
    merged = _page_align(ranges, 512 * 4096)
    assert len(merged) == 3, merged
    assert merged[0] == (21 * 4096, 171 * 4096)


def test_alignment_rounds_outward_never_inward():
    """Ein paar Byte zu viel zu pinnen kostet eine Seite; zu wenig laesst die
    GPU auf ungepinnte Seiten zeigen."""
    out = _page_align([(100, 5000)], 1 << 20)
    assert out == [(0, 8192)]


def test_no_ranges_means_the_whole_file_as_before():
    """Ohne `rows` bleibt alles wie vor #84 -- ein Aufrufer ohne
    Zeilenwissen darf nichts kaputt machen."""
    d = tempfile.mkdtemp(dir="/dev/shm")
    t, _ = es.open_store(d, "L1", "w13", 8, (4096,), torch.uint8,
                         register=False)
    assert t.shape == (8, 4096)


def test_rows_outside_the_file_are_dropped():
    """Eine Zeile jenseits der Datei waere eine Registrierung ins Leere."""
    d = tempfile.mkdtemp(dir="/dev/shm")
    t, _ = es.open_store(d, "L2", "w13", 4, (4096,), torch.uint8,
                         register=False, rows=[0, 99, -1])
    assert t.shape == (4, 4096)
