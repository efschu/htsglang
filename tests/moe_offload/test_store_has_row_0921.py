"""#75 (21.09.): der Sentinel-Leser, den es nicht gab.

`mark_rows_written` publiziert seit dem ersten Tag, welche Store-Zeilen ein
Rang geschrieben hat. `rows_written` liest das -- und hatte NULL AUFRUFER
(devindex, 21.09.). Gruppe D liest deshalb den ganzen Checkpoint ein zweites
Mal, obwohl P dieselben Experten im geteilten tmpfs hat: fnFL2w1 237 s und
~14 GiB Page-Cache gegen eine Container-Decke von ~93 GiB.

Dieser Test nagelt die Semantik des Lesers fest, BEVOR ein Verbraucher daran
haengt -- konservativ in jede unsichere Richtung, weil der Irrtum "lies noch
einmal" Zeit kostet und der Irrtum "ueberspring es" ein Modell mit einer
Luecke laedt.

OFFEN, und hier festgehalten, damit es niemand uebersieht: der Store haelt
REPACKTE Bytes (`[presplit] layer N: repack + presplit at load`), der
Checkpoint rohe. Ein Veto im Ladepfad nach Checkpoint-NAMEN kann die beiden
daher nicht aufeinander abbilden; der Verbraucher dieses Lesers muss NACH dem
Modellbau ansetzen und die Post-Repack-Tensoren aus dem Store fuellen.
"""

import os
import tempfile

from sglang.srt.layers.moe import expert_store as es


def test_store_has_row_is_conservative(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv(es.STORE_DIR_ENV, d)
    es.forget_written_rows()

    # kein Sentinel -> "lies vom Checkpoint"
    assert es.store_has_row("L0", "w13_qweight", 5, 3) is False

    es.mark_rows_written(d, "L0", "w13_qweight", 0, [3, 5, 7])
    assert es.store_has_row("L0", "w13_qweight", 5, 3) is True
    # eine Zeile, die niemand geschrieben hat
    assert es.store_has_row("L0", "w13_qweight", 6, 3) is False
    # ein anderes Tensor-Attribut teilt den Sentinel NICHT
    assert es.store_has_row("L0", "w2_qweight", 5, 3) is False

    # der Cache darf ein frisches Publish nicht verpassen
    es.mark_rows_written(d, "L0", "w13_qweight", 1, [6])
    assert es.store_has_row("L0", "w13_qweight", 6, 3) is True


def test_store_off_answers_false(monkeypatch):
    """Ohne Store gibt es nichts zu ueberspringen."""
    monkeypatch.setenv(es.STORE_DIR_ENV, "")
    es.forget_written_rows()
    assert es.store_has_row("L0", "w13_qweight", 5, 3) is False


def test_an_unreadable_sentinel_loads_rather_than_raises(monkeypatch):
    """Ein kaputter Sentinel ist ein Lader, kein Absturz -- der Checkpoint
    liegt ja noch da."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv(es.STORE_DIR_ENV, d)
    es.forget_written_rows()
    path = es.store_path(d, "L0", "w13_qweight") + ".r0.written.json"
    with open(path, "w") as fh:
        fh.write("{ this is not json")
    assert es.store_has_row("L0", "w13_qweight", 5, 3) is False


def test_the_reader_is_exported():
    """Ein Leser, den man nicht importieren kann, bleibt wieder ohne
    Aufrufer -- genau der Zustand, den #75 beendet."""
    for name in ("rows_written", "written_rows_cached",
                 "forget_written_rows", "store_has_row"):
        assert name in es.__all__, name
