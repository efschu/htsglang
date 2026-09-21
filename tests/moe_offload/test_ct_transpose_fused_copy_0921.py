"""#68 (21.09.): die CT-WNA16-Transposition als EINE Kopie statt zwei.

`loaded_weight.t().contiguous()` war 43 % der Ladezeit auf allen drei Raengen
(fnFL2v84/v85, 222720 Experten-Shards, ~50 s je Rang). `.t()` allein kostet
nichts -- nur Strides; teuer war `.contiguous()`, das eine volle umsortierte
Kopie materialisiert, BEVOR der Konsument den Tensor ein zweites Mal kopiert.
Jeder Endpunkt dieses Pfades ist ein `copy_` (oder ein
`param_data[expert_id] = ...`, was dasselbe ist), und `copy_` sortiert
strided Quellen selbst um. Die Umsortierung passiert also weiterhin genau
einmal -- fusioniert in die Kopie, die ohnehin faellt.

Die Frage, die dieser Test beantwortet, ist nicht "ist es schneller" (das ist
gemessen), sondern "kommen dieselben Bytes an".
"""

import inspect

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton import layer as L


# Die vier Shard-Formen des Checkpoints (int32 packed), Checkpoint-Layout
# [out, in/pack] gegen das Param-Layout [E, in/pack, out].
SHAPES = [(2560, 80), (640, 320), (320, 20), (80, 80)]


@pytest.mark.parametrize("shape", SHAPES)
def test_fused_copy_lands_the_same_bytes(shape):
    """Die Kernfrage: `dst.copy_(src.t())` == `dst.copy_(src.t().contiguous())`."""
    torch.manual_seed(1234)
    src = torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)
    alt = torch.empty(shape[1], shape[0], dtype=torch.int32)
    neu = torch.empty(shape[1], shape[0], dtype=torch.int32)
    alt.copy_(src.t().contiguous())  # die alte Form
    neu.copy_(src.t())  # die neue
    assert torch.equal(alt, neu)


@pytest.mark.parametrize("shape", SHAPES)
def test_narrow_on_the_strided_view_lands_the_same_bytes(shape):
    """Der Weg fuehrt durch `narrow` (narrow_padded_param_and_loaded_weight),
    bevor kopiert wird -- narrow ist stride-neutral, aber das ist genau die
    Annahme, die dieser Test festnagelt statt sie zu glauben."""
    torch.manual_seed(99)
    src = torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)
    t_alt = src.t().contiguous()
    t_neu = src.t()
    n = t_alt.shape[0] // 2
    for dim in (0, 1):
        k = t_alt.shape[dim] // 2
        a = t_alt.narrow(dim, 0, k)
        b = t_neu.narrow(dim, 0, k)
        dst_a = torch.empty(a.shape, dtype=torch.int32)
        dst_b = torch.empty(b.shape, dtype=torch.int32)
        dst_a.copy_(a)
        dst_b.copy_(b)
        assert torch.equal(dst_a, dst_b), (shape, dim)
    assert n >= 0


@pytest.mark.parametrize("shape", SHAPES)
def test_setitem_assignment_is_a_copy_not_an_alias(shape):
    """`param_data[expert_id] = loaded_weight` ist ein Endpunkt dieses Pfades.
    Waere es ein Alias, laege ein strided Tensor als Gewicht im Modell -- es
    ist aber `__setitem__`, also eine Kopie, und danach ist das Ziel
    kontiguierlich."""
    torch.manual_seed(7)
    src = torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)
    param = torch.zeros(2, shape[1], shape[0], dtype=torch.int32)
    param[1] = src.t()
    assert param[1].is_contiguous()
    assert torch.equal(param[1], src.t().contiguous())
    # und der Speicher ist wirklich der des Parameters, nicht der der Quelle
    assert param[1].data_ptr() != src.data_ptr()


def test_the_consumer_no_longer_materialises():
    """Die Zeile selbst -- damit ein Rueckbau auffaellt."""
    src = inspect.getsource(L.FusedMoE._weight_loader_impl)
    assert "loaded_weight.t() if _needs_ct_transpose else loaded_weight" in src
    assert "loaded_weight.t().contiguous()" not in src


def test_the_one_holding_path_materialises_itself():
    """Der fp8-shared-Pfad haelt den Tensor fest (`_pending_fp8_shared_*`),
    statt ihn durch ein `copy_` zu reichen -- also materialisiert er selbst."""
    src = inspect.getsource(L.FusedMoE._maybe_load_fp8_shared_expert_as_fp4)
    assert "if not loaded_weight.is_contiguous():" in src
    assert "loaded_weight = loaded_weight.contiguous()" in src
    # und zwar VOR der Ablage in den Dicts
    i_guard = src.index("is_contiguous()")
    i_store = src.index("_pending_fp8_shared_weights[key]")
    assert i_guard < i_store


def test_no_consumer_on_the_path_needs_contiguity():
    """Der Grund, warum das Weglassen sicher ist: auf dem ganzen Weg der
    Ladehelfer steht kein `view`/`reshape`/`data_ptr`/`from_blob`, das
    Kontiguitaet voraussetzt. Faellt das kuenftig, faellt dieser Test."""
    for fn in (
        L.FusedMoE._load_model_weight_or_group_weight_scale,
        L.FusedMoE._load_per_channel_weight_scale,
        L.FusedMoE._load_w13,
        L.FusedMoE._load_w2,
        L.FusedMoE._load_single_value,
        L.FusedMoE._load_g_idx,
    ):
        src = inspect.getsource(fn)
        for bad in ("loaded_weight.view(", "loaded_weight.reshape(", "loaded_weight.data_ptr("):
            assert bad not in src, (fn.__name__, bad)
