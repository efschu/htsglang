"""#80 (fnFL2w7, 21.09.): ein Nicht-Halter ist kein Ziel, auch wenn die Form
REPLIZIERT heisst.

Seit #76 darf eine Teilmenge der TP-Gruppe einen Tensor halten, solange alle
Halter ihn GANZ tragen -- unter Form A ist das der Normalfall: der
Attention-Host haelt jedes Dense-Gewicht, die Experten-Worker keins. Der
Join traegt das als Breitenvektor (w, 0, 0) ein.

Dieser Vektor erreichte `_blocks_of` nie: `geom()` setzte `dst_widths` nur
bei `self.sharded`, und REPLICATED ist nicht sharded. Also fiel der Plan auf
den gleichmaessigen Split ueber ALLE Raenge zurueck und adressierte Karten,
die den Tensor gar nicht haben.

GEMESSEN an fnFL2w7:
    lane=c1 descs=37  slot_bytes=81.466.808
    lane=c2 descs=113 slot_bytes=237.381.840
    beide scheitern an layers.29/42.attn_hyper_connection.block_inject_weight
und die Manifeste desselben Boots sagen:
    D rank0 = 384 solche Tensoren, D rank1 = 0, D rank2 = 0.
P lief in sein volles Lane-Budget, dann W29 auf resume_memory_occupation,
0/6 Raenge.
"""

from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import xchg_manifest as xm


def _joined(widths, axis, name="model.layers.0.attn_hyper_connection.w"):
    return xm.JoinedTensor(
        param_name=name, tensor_class="Linear", tag="weights_0", itemsize=2,
        rows_full=sum(widths) or 4, cols_full=8, shard_axis=axis,
        pp_stage=0, tp_widths=tuple(widths), pp_card=0,
    )


def test_a_replicated_solo_holder_carries_its_zero_widths():
    """Der gemessene Form-A-Fall: nur Rang 0 haelt, die Breiten sagen es,
    und sie muessen bis zum Plan durchkommen."""
    t = _joined((4, 0, 0), wx.REPLICATED)
    assert t.geom(tp_is_dst=True).dst_widths == (4, 0, 0)


def test_a_replicated_tensor_every_rank_holds_stays_unchanged():
    """Die Gegenrichtung: ohne Null-Breite aendert der Fix nichts -- ein
    echtes Replikat bleibt ein Replikat ohne dst_widths."""
    t = _joined((4, 4, 4), wx.REPLICATED)
    assert t.geom(tp_is_dst=True).dst_widths is None


def test_an_ordinary_cut_still_carries_its_widths():
    """Der sharded Pfad war nie kaputt und bleibt, wie er war."""
    t = _joined((2, 1, 1), wx.ROWS)
    assert t.geom(tp_is_dst=True).dst_widths == (2, 1, 1)


def test_mixed_fused_stays_exempt():
    """MIXED_FUSED liest dst_widths nicht und seine Aussenbreiten summieren
    sich nicht auf rows_full -- der Fix darf es nicht hineinziehen, sonst
    verweigert validate() eine gesunde Geometrie.

    Am QUELLTEXT geprueft statt an einer gebauten Geometrie: eine
    MIXED_FUSED-Geometrie aus synthetischen Breiten faellt schon in
    `validate()`, also wuerde der Bau die Ausnahme verdecken statt sie zu
    zeigen."""
    import inspect

    src = inspect.getsource(xm.JoinedTensor.geom)
    i_mixed = src.index("self.shard_axis != wx.MIXED_FUSED")
    i_zero = src.index("any(int(w) == 0 for w in self.tp_widths)")
    assert i_mixed < i_zero, (
        "die MIXED_FUSED-Ausnahme muss VOR der Null-Breiten-Klausel stehen "
        "und sie per `and` binden")


def test_the_source_side_is_untouched():
    """dst_widths ist eine ZIEL-Aussage; die Quelle liest sie nicht."""
    t = _joined((4, 0, 0), wx.REPLICATED)
    assert t.geom(tp_is_dst=False).dst_widths is None
