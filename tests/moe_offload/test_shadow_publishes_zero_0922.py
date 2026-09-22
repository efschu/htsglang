"""#103: ein Schatten haelt nichts -- aber er EXISTIERT.

fnFL2w38, 05:15:36Z: der Flip stirbt auf allen drei D-Raengen an

    W106 Weg2XchgWakeSourceGapRefused: group=D rank=0 tag=weights_draft
    expected_bytes=5593104384: weights_cpu_backup_armed()=False, the
    exchange's own join carries zero descriptors for it on this leg, AND
    the disk-reload fallback (#1394) is undefined on this
    'compressed-tensors' checkpoint -- No third net exists for this tag

Der Draft KANN aber D2D gehen, beide Phasen halten ihn (gemessen, w38):

    D rank=0 card=0 region_tag=weights_draft pieces=34 bytes=4139515392
    P rank=2 card=2 region_tag=weights_draft pieces=34 bytes=4139515392

Verschiedene Karten, identische Bytes -- also BAR1 DMA, kein Host-Backup
und kein Disk-Reload. Es scheitert allein daran, dass der Join nur EINE
Karte sieht: die beiden Experten-Worker sind meta-Schatten, und #76
(21.09., mein eigener Fix) laesst sie GAR KEIN Manifest schreiben:

    WEG2-XCHG-MANIFEST-WRITE group=D rank=1 region_tag=weights_draft
      shadow_pieces=19 of 19 dropped -- meta tensors hold no bytes and
      are not published as holders (#76)
    WEG2-XCHG-MANIFEST-WRITE group=D rank=1 pieces=0
      reason=all-meta-shadow -- no manifest written

#76 war richtig (ein Schatten traegt die UNREPACKTE Form und widerspricht
dem Host), ging aber zu weit: aus "haelt nichts" wurde "existiert nicht".
Der Join zaehlt dann `len(tp)==1`, `refuse_diagonal_layout` wirft W68,
und der Tag bekommt null Descriptors.

Ein leeres Manifest sagt beides zugleich: der Rang IST da, und er haelt
NICHTS -- also Breite 0, was #102s Halter-Karte genau liest.
"""

import pytest

from sglang.srt.weg2 import xchg_manifest as xm


def _piece(name, rows=4096, cols=1024):
    return xm.ManifestPiece(param_name=name, tensor_class="LinearBase",
                            rows_full=rows, cols_full=cols, itemsize=2,
                            tag="weights_draft", nbytes=rows * cols * 2)


def _man(group, rank, card, pieces):
    return xm.RankManifest(group=group, rank=rank, card=card,
                           region_tag="weights_draft",
                           boot_token="t", pieces=tuple(pieces))


NAME = "model.layers.0.mtp.fc.weight"


def test_leeres_manifest_haelt_den_rang_im_join():
    """DER FALL, DER w38 TOETETE: nur Rang 0 haelt den Draft."""
    mans = [
        _man("D", 0, 0, [_piece(NAME)]),   # der Halter
        _man("D", 1, 1, []),               # Schatten: da, haelt nichts
        _man("D", 2, 2, []),               # Schatten
        _man("P", 2, 2, [_piece(NAME)]),   # P haelt ihn auf Karte 2
    ]
    join = xm.join_manifests(mans, pp_group="P", tp_group="D")
    assert len(join.cards) == 3, (
        "drei Raenge, drei Karten -- mit nur einer Karte haelt "
        "refuse_diagonal_layout den Austausch fuer die PP-Form und wirft W68")
    t = next(t for t in join.tensors if t.param_name == NAME)
    assert t.tp_widths[0] > 0, "Rang 0 haelt den Tensor"
    assert t.tp_widths[1] == 0 and t.tp_widths[2] == 0, (
        "die Schatten halten NICHTS -- Breite 0, nicht 'fehlt'")


def test_ohne_die_schatten_sieht_der_join_nur_eine_karte():
    """Die Gegenprobe: genau so sieht es heute aus, und genau das bricht."""
    mans = [
        _man("D", 0, 0, [_piece(NAME)]),
        _man("P", 2, 2, [_piece(NAME)]),
    ]
    join = xm.join_manifests(mans, pp_group="P", tp_group="D")
    assert len(join.cards) == 1, (
        "ohne die Schatten-Manifeste bleibt eine Karte uebrig -- das ist "
        "der w38-Zustand, und er fuehrt in refuse_diagonal_layout")
