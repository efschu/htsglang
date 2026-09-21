"""#72 Ringpuffer, Stufe 1: der Store gibt eine Zeile wieder her.

Nutzer 21.09.: "die anderen liegen im vram, es ist quasi ein ringpuffer".
Damit ist der Store ein Cache ueber dem Checkpoint -- und einem Cache fehlte
bis hier die Rueckgabe: `write_rows` fuellt, nichts leerte.

Zwei Eigenschaften sind hier festgenagelt, beide aus der Physik dieser Box:

1. DIE REIHENFOLGE. Erst den Sentinel zuruecknehmen, dann lochen. Andersherum
   gibt es ein Fenster, in dem der Sentinel eine Zeile verspricht, die schon
   Nullen ist -- und Nullen sind das eine falsche Ergebnis, das kein Leser als
   Fehler erkennt (er rechnet einfach mit einem leeren Experten weiter).
2. DIE VERWEIGERUNG AUF EINER REGISTRIERTEN REGION. `shared_pinned_empty`
   registriert die GANZE Datei fuer die GPU (shared_pinned.py:91), und
   `cudaHostUnregister` loest nur die ganze Region. Seiten unter einer
   lebenden Device-Mapping freizugeben ist kein Speichergewinn, sondern
   xsn277 ('illegal memory access') oder stille Korruption.
"""

import json
import os
import tempfile

import pytest

from sglang.srt.layers.moe import expert_store as es

ROW = 4096  # eine Seite je Zeile, damit das Loch sichtbar wird


def _store(tmpdir, rows=8):
    path = es.store_path(tmpdir, "L0", "w13_qweight")
    with open(path, "wb") as fh:
        # ZUFALLSBYTES, NICHT b"\xab"*n: der Store liegt auf ZFS mit
        # Kompression, und eine Datei aus lauter gleichen Bytes belegt dort
        # 512 Byte statt 32 KiB -- dann misst `st_blocks` das Loch nicht,
        # weil vorher schon nichts belegt war. Der erste Lauf dieses Tests
        # scheiterte genau daran (before=after=512), am Messwerkzeug, nicht
        # am Code.
        fh.write(os.urandom(rows * ROW))
    return path


def _allocated_bytes(path):
    """Was die Datei WIRKLICH belegt -- st_blocks, nicht st_size. Genau diese
    Differenz ist der ganze Sinn des Lochs."""
    return os.stat(path).st_blocks * 512


def test_punching_frees_the_pages(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv(es.STORE_DIR_ENV, d)
    path = _store(d)
    before = _allocated_bytes(path)
    freed = es.punch_rows(d, "L0", "w13_qweight", [2, 3, 4], ROW)
    after = _allocated_bytes(path)
    assert freed == 3 * ROW
    assert before - after >= 3 * ROW, (before, after)
    # die Dateigroesse bleibt -- die Zeilen danach behalten ihren Platz
    assert os.stat(path).st_size == 8 * ROW


def test_the_sentinel_is_withdrawn_before_the_hole(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv(es.STORE_DIR_ENV, d)
    _store(d)
    es.forget_written_rows()
    es.mark_rows_written(d, "L0", "w13_qweight", 0, [1, 2, 3])
    assert es.store_has_row("L0", "w13_qweight", 2, 1) is True

    es.punch_rows(d, "L0", "w13_qweight", [2], ROW, rank=0)

    assert es.store_has_row("L0", "w13_qweight", 2, 1) is False, \
        "eine gelochte Zeile darf der Sentinel nicht mehr versprechen"
    # die Nachbarn bleiben versprochen
    assert es.store_has_row("L0", "w13_qweight", 1, 1) is True
    assert es.store_has_row("L0", "w13_qweight", 3, 1) is True


def test_a_registered_store_refuses_rather_than_corrupts(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv(es.STORE_DIR_ENV, d)
    path = _store(d)
    before = _allocated_bytes(path)
    with pytest.raises(es.StorePinnedForDevice):
        es.punch_rows(d, "L0", "w13_qweight", [2], ROW, registered=True)
    assert _allocated_bytes(path) == before, "verweigert heisst: nichts getan"


def test_contiguous_rows_become_one_hole(monkeypatch):
    """210 Einzelloecher fuer dieselben Seiten waeren 210 Baumoperationen."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv(es.STORE_DIR_ENV, d)
    _store(d, rows=16)
    calls = []
    real = es._fallocate_punch
    monkeypatch.setattr(es, "_fallocate_punch",
                        lambda fd, off, ln: (calls.append((off, ln)),
                                             real(fd, off, ln))[1])
    es.punch_rows(d, "L0", "w13_qweight", [1, 2, 3, 9, 10, 15], ROW)
    assert [ln // ROW for _, ln in calls] == [3, 2, 1], calls


def test_unmarking_what_was_never_marked_is_not_an_error(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv(es.STORE_DIR_ENV, d)
    _store(d)
    es.forget_written_rows()
    es.unmark_rows_written(d, "L0", "w13_qweight", 0, [5])
    assert es.store_has_row("L0", "w13_qweight", 5, 1) is False
    with open(es._sentinel(d, "L0", "w13_qweight", 0)) as fh:
        assert json.load(fh)["rows"] == []


def test_punching_a_store_that_does_not_exist_is_a_no_op(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv(es.STORE_DIR_ENV, d)
    assert es.punch_rows(d, "L0", "w13_qweight", [1], ROW) == 0


def test_the_eviction_is_exported():
    """Ein Werkzeug ohne Verdrahtung ist der Defekt, den #75 aufgedeckt hat --
    `rows_written` hatte monatelang null Aufrufer."""
    for name in ("punch_rows", "unmark_rows_written", "StorePinnedForDevice"):
        assert name in es.__all__, name
