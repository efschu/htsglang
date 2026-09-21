"""#76 (21.09.): ein meta-Schatten haelt nichts, also publiziert er nichts --
und wer eine Region nicht fuehrt, ist kein fehlender Halter.

fnFL2w2 erreichte den ersten Flip und starb am Wake mit

    W68 Weg2XchgPlanDisagree: model.layers.0.mlp.experts.w13_weight_packed:
    the PP side holds (81920, 2560) and the TP rows hold
    [(81920, 2560), (163840, 1280), (163840, 1280)]

Beide Formen tragen DIESELBEN 838.860.800 Bytes. Die Erklaerung steht im
Boot-Log desselben Boots: unter Form A sind die Experten-Worker
Draft-SCHATTEN ("rank 1 is a draft SHADOW (meta-device draft, no draft
weights/KV/graphs)"), und der Census bestaetigt es mit
`weights_draft mib=0.0`. Ein meta-Tensor durchlaeuft
`process_weights_after_loading` nie, traegt also die UNREPACKTE Form -- und
das Manifest publizierte ihn trotzdem als Halter von 2,54 GiB.

Beide Haelften gehoeren zusammen: nimmt man nur die Schatten aus dem
Manifest, kippt W68 in W74 (offline an den echten Manifesten gemessen:
1603 von 1853). Denn der Join verlangte bis hier, dass JEDER TP-Rang JEDEN
Tensor haelt -- was unter Form A nie gilt, wo der Attention-Host alle Dense-
Gewichte allein traegt.
"""

import pytest

from sglang.srt.weg2 import xchg_manifest as xm
from sglang.srt.weg2 import weight_exchange as wx


class _FakeTensor:
    def __init__(self, kind):
        self.device = type("D", (), {"type": kind})()


def test_a_meta_tensor_is_not_a_holder():
    from sglang.srt.weg2.weight_exchange import _tensor_is_meta

    assert _tensor_is_meta(_FakeTensor("meta")) is True
    assert _tensor_is_meta(_FakeTensor("cuda")) is False
    assert _tensor_is_meta(_FakeTensor("cpu")) is False


def test_what_it_cannot_read_stays_in_the_manifest():
    """Die Irrtumsrichtung: ein Tensor, den dieser Test nicht lesen kann,
    bleibt drin. Ihn fallen zu lassen wuerde den Austausch still verengen;
    ihn zu behalten faellt spaetestens im Join auf."""
    from sglang.srt.weg2.weight_exchange import _tensor_is_meta

    assert _tensor_is_meta(object()) is False
    assert _tensor_is_meta(None) is False


def _piece(name, rows, cols, tag="weights_0", item=4):
    return xm.ManifestPiece(
        param_name=name, tensor_class="CompressedTensorsWNA16MoE",
        rows_full=int(rows), cols_full=int(cols), itemsize=int(item),
        tag=tag, nbytes=int(rows) * int(cols) * int(item),
    )


def _man(group, rank, pieces, region="weights"):
    return xm.RankManifest(
        group=group, rank=rank, card=rank, region_tag=region,
        boot_token="t", pieces=tuple(pieces), tp_rank=rank, pp_rank=0,
    )


DENSE = "model.layers.0.self_attn.qkv_proj.weight_packed"
DRAFT = "model.layers.0.mlp.experts.w13_weight_packed"


def test_form_a_dense_lives_on_the_host_alone_and_still_joins():
    """Der gemessene Form-A-Fall: nur D-Rang 0 haelt das Dense-Gewicht, die
    Experten-Worker tragen dort HostOnlyModule-Platzhalter."""
    pp = [_man("P", 0, [_piece(DENSE, 100, 8)])]
    tp = [
        _man("D", 0, [_piece(DENSE, 100, 8), _piece("x.norm.weight", 1, 8)]),
        _man("D", 1, [_piece("x.norm.weight", 1, 8)]),
        _man("D", 2, [_piece("x.norm.weight", 1, 8)]),
    ]
    j = xm.join_manifests(pp + tp)
    got = {t.param_name: t for t in j.tensors}
    assert DENSE in got, "der Solo-Halter muss geplant werden, nicht verweigert"
    t = got[DENSE]
    assert t.shard_axis == wx.REPLICATED
    # Der Breitenvektor bleibt in RANGORDNUNG: wer nichts haelt, haelt null.
    assert len(t.tp_widths) == 3 and t.tp_widths[1] == 0 and t.tp_widths[2] == 0


def test_a_missing_piece_of_a_REAL_CUT_is_still_refused():
    """Die Gefahrenrichtung, die bleiben muss: fehlt einem Rang sein Stueck
    eines echten Schnitts, verschoebe ein Plan ueber die uebrigen Raenge jede
    Shard-Grenze. Das bleibt unsourced."""
    pp = [_man("P", 0, [_piece(DENSE, 300, 8)])]
    tp = [
        _man("D", 0, [_piece(DENSE, 100, 8)]),
        _man("D", 1, [_piece(DENSE, 100, 8)]),   # Rang 2 fehlt sein Drittel
        _man("D", 2, [_piece("x.norm.weight", 1, 8)]),
    ]
    with pytest.raises(wx.Weg2XchgSourceMissing):
        xm.join_manifests(pp + tp)


def test_a_region_no_rank_but_one_leads_is_not_a_gap():
    """Die Draft-Region nach dem Schatten-Schnitt: nur D-Rang 0 fuehrt sie
    ueberhaupt. Das ist keine Luecke, sondern die Rollenverteilung."""
    dtag = wx.GPU_MEMORY_TYPE_WEIGHTS_DRAFT
    pp = [_man("P", 2, [_piece(DRAFT, 81920, 2560, tag=dtag)], region=dtag),
          _man("P", 0, [_piece(DENSE, 10, 8)]),
          _man("P", 1, [_piece("y.norm.weight", 1, 8)])]
    tp = [
        _man("D", 0, [_piece(DENSE, 10, 8), _piece("y.norm.weight", 1, 8)]),
        _man("D", 1, [_piece("y.norm.weight", 1, 8)]),
        _man("D", 2, [_piece("y.norm.weight", 1, 8)]),
        _man("D", 0, [_piece(DRAFT, 81920, 2560, tag=dtag)], region=dtag),
    ]
    j = xm.join_manifests(pp + tp)
    assert DRAFT in {t.param_name for t in j.tensors}


def test_the_measured_shadow_shapes_no_longer_meet_in_the_join():
    """Die Zahlen des Boots: haette der Schatten publiziert, stuenden die
    beiden Formen wieder nebeneinander. Dieser Test haelt fest, dass genau
    diese Paarung ein Zwist IST -- der Fix beseitigt den Zwist, indem der
    Schatten gar nicht erst Halter wird, nicht indem er ihn wegdeutet."""
    whole = _piece(DRAFT, 81920, 2560)
    cut = [_piece(DRAFT, 81920, 2560), _piece(DRAFT, 163840, 1280),
           _piece(DRAFT, 163840, 1280)]
    with pytest.raises(wx.Weg2XchgPlanDisagree):
        xm._axis_of(DRAFT, whole, cut)
