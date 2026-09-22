"""Die Halter-Karte gilt nur fuer das ZIEL -- die Quelle shardet blind.

fnFL2w36/w37, Gruppe D als QUELLE (direction=tp_to_pp), Form A
(`--rank-tp-ratio 1,0,0`: alles Dense auf Rang 0, die 3080er sind reine
Experten-Worker):

    W68 Weg2XchgPlanDisagree: lane src=1 dst=1 tag='weights_9':
    33 of 37 descs have no address on the side this rank owns
    (first: model.layers.29.attn_hyper_connection.block_inject_weight.weight)

D TP1s Adressbuch fuehrt 600 Namen, ALLE `mlp.experts.*`; TP0s Manifest
2011 Eintraege gegen 408. Die Dense-Tensoren liegen ganz auf Rang 0 --
der Plan verlangt sie trotzdem von Rang 1, weil `_blocks_of` den
per-Tensor-Breitenvektor nur auf der Zielseite liest:

    if is_dst and geom.dst_widths is not None:

Der Kommentar bei #1378 xsn54 sagt es selbst: "the SOURCE side keeps its
every-rank shape until its own defect is measured". Gemessen ist er
jetzt: zwei Boots, beide Male alle drei P-Raenge tot, weil D die Lane
nie deponierte und P 90 s ins Zeitbudget lief.
"""

import pytest

from sglang.srt.weg2 import weight_exchange as wx


def _form_a_layout():
    """Gruppe D: drei Karten, TP-Form -- so plant der Join sie heute."""
    return wx.GroupLayout(name="D", cards=(0, 1, 2), tp_size=3)


def _dense_geom(**kw):
    """Ein Dense-Tensor, wie Layer 29 ihn stellt: geshardete Achse, aber
    unter Form A traegt ihn NUR Rang 0 (Breiten 4096,0,0)."""
    return wx.ParamGeom(
        name="model.layers.29.linear_attn.out_proj.weight_packed",
        tag="weights_9", shard_axis=0,
        rows_full=4098, cols_full=1024, itemsize=2, **kw)


def test_quelle_ehrt_die_halter_karte(monkeypatch):
    """DER FALL, DER w36 UND w37 TOETETE."""
    geom = _dense_geom(src_widths=(4098, 0, 0))
    blocks = wx._blocks_of(geom, _form_a_layout(), is_dst=False)
    assert len(blocks) == 3, "drei Raenge, drei Blocklisten"
    assert sum(b.size for b in blocks[0]) == 4098, (
        "Rang 0 HAELT den Tensor ganz")
    assert sum(b.size for b in blocks[1]) == 0, (
        "Rang 1 haelt unter Form A kein Dense -- bekommt er hier Bytes, "
        "verlangt die Lane sie spaeter von ihm (w37: 33 of 37 descs have "
        "no address on the side this rank owns)")
    assert sum(b.size for b in blocks[2]) == 0, "Rang 2 ebenso"


def test_ziel_bleibt_unveraendert(monkeypatch):
    """Die Gegenrichtung P->D darf sich NICHT aendern (dst_widths seit je)."""
    geom = _dense_geom(dst_widths=(4098, 0, 0))
    blocks = wx._blocks_of(geom, _form_a_layout(), is_dst=True)
    assert sum(b.size for b in blocks[0]) == 4098
    assert sum(b.size for b in blocks[1]) == 0
    assert sum(b.size for b in blocks[2]) == 0


def test_ohne_karte_shardet_die_quelle_weiter(monkeypatch):
    """Ohne Breitenvektor bleibt der generische Split -- sonst prueft der
    erste Test nur, dass irgendetwas leer ist."""
    geom = _dense_geom()
    blocks = wx._blocks_of(geom, _form_a_layout(), is_dst=False)
    assert all(sum(b.size for b in r) > 0 for r in blocks), (
        "ohne Halter-Karte teilt der Plan weiter ueber alle Raenge -- diese "
        "Form ist die richtige fuer eine echte TP-Gruppe")


def _tensor(widths, shard_axis=0):
    """Ein XchgTensor, wie der Join ihn baut."""
    from sglang.srt.weg2 import xchg_manifest as xm
    return xm.JoinedTensor(
        param_name="model.layers.29.linear_attn.out_proj.weight_packed",
        tensor_class="LinearBase", tag="weights_9", itemsize=2,
        rows_full=4098, cols_full=1024, shard_axis=shard_axis,
        pp_stage=1, tp_widths=tuple(widths), pp_card=1)


def test_geom_traegt_die_karte_in_beide_richtungen():
    """Der Join muss die Breiten auf DER Seite mitgeben, die gerade Quelle ist."""
    t = _tensor((4098, 0, 0))
    g_src = t.geom(tp_is_dst=False)   # D->P: die TP-Gruppe ist QUELLE
    g_dst = t.geom(tp_is_dst=True)    # P->D: die TP-Gruppe ist ZIEL
    assert g_src.src_widths == (4098, 0, 0), (
        "ohne diese Karte verlangt der Plan Dense von einem Rang, der es "
        "nicht haelt -- der w36/w37-Tod")
    assert g_src.dst_widths is None, "als Quelle ist die Zielkarte nicht dran"
    assert g_dst.dst_widths == (4098, 0, 0), "die Gegenrichtung bleibt, wie sie war"
    assert g_dst.src_widths is None
