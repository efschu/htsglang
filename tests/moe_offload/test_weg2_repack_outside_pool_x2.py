"""fnFL2x2 (22.09.): der Experten-Repack laeuft AUSSERHALB des Tag-Pools, nur
die Ueberlebenden werden darin geboren.

Gemessen an x2 nach dem Laden, Basis-Pool `weights` je P-Stufe: 15,15 / 8,85 /
7,04 GiB aktiv neben 5,46 / 4,76 / 4,21 GiB INAKTIV (tote Segmente des
Repacks), Draft-Pool 1,35 GiB inaktiv. Stufe 2 fehlten 181 MiB fuer den ersten
KV-Token. `release_active_tag_pools` hat in 48 von 48 Aufrufen nichts
freigegeben: ein lebender MemPool hat use_count >= 1.

Die Probe am Metall (pool_nest_probe.py, 3080, torch 2.11): raus, 600 MiB
Transient, wieder rein, 100 MiB Ueberlebender, raus, Transient frei ->
empty_cache laesst 400 MiB reserviert, der Pool haelt genau seine 450 MiB.
Hier hermetisch: die Reihenfolge der begin/end-Aufrufe.
"""

import types

import pytest
import torch

from sglang.srt.managers import weg2_memory_saver as MS


class _FakePool:
    def __init__(self, ident=(0, 7)):
        self.id = ident


@pytest.fixture
def stepped(monkeypatch):
    calls = []
    monkeypatch.setattr(MS, "_ACTIVE_TAG_POOL", _FakePool(), raising=False)
    monkeypatch.setattr(MS, "_STEPPED_OUT_POOLS", [], raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    mod = types.ModuleType("torch.cuda.memory")
    mod._cuda_endAllocateToPool = lambda dev, pid: calls.append("end")
    mod._cuda_beginAllocateCurrentThreadToPool = lambda dev, pid: calls.append(
        "begin"
    )
    monkeypatch.setitem(__import__("sys").modules, "torch.cuda.memory", mod)
    return calls


def test_back_into_ohne_ausstieg_ist_ein_no_op(stepped):
    with MS.back_into_tag_pool() as drin:
        assert drin is False
    assert stepped == []


def test_ueberlebender_wird_im_pool_geboren(stepped):
    with MS.outside_tag_pool(reason="x"):
        assert stepped == ["end"]
        with MS.back_into_tag_pool() as drin:
            assert drin is True
            assert stepped == ["end", "begin"]
        assert stepped == ["end", "begin", "end"]
    assert stepped == ["end", "begin", "end", "begin"]
    assert MS._STEPPED_OUT_POOLS == []


def test_verschachteltes_outside_ruehrt_den_allokator_nicht_an(stepped):
    """Ein zweites end fuer einen schon verlassenen Pool wuerde einen fremden
    Capture-Eintrag entfernen -- der ct-stream-Wrapper und der Schema-Wrapper
    liegen genau so ineinander."""
    with MS.outside_tag_pool(reason="ct-stream"):
        with MS.outside_tag_pool(reason="ct-moe-repack") as out:
            assert out is True
        assert stepped == ["end"]
        with MS.back_into_tag_pool():
            pass
    assert stepped == ["end", "begin", "end", "begin"]


def test_ausnahme_im_ueberlebenden_stellt_den_zustand_wieder_her(stepped):
    with pytest.raises(RuntimeError):
        with MS.outside_tag_pool():
            with MS.back_into_tag_pool():
                raise RuntimeError("alloc blew up")
    assert stepped == ["end", "begin", "end", "begin"]
    assert MS._STEPPED_OUT_POOLS == []


def test_schema_wrapper_steigt_aus_und_der_repack_steigt_fuer_ueberlebende_ein():
    import inspect

    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa16_moe as moe,
    )

    cls = moe.CompressedTensorsWNA16MoE
    outer = inspect.getsource(cls.process_weights_after_loading)
    assert 'outside_tag_pool(reason="ct-moe-repack")' in outer
    assert "self._repack_to_marlin(layer)" in outer
    body = inspect.getsource(cls._repack_to_marlin)
    # Workspace und resize_ (waechst er, ist der neue Speicher ein Ueberlebender)
    assert body.count("with back_into_tag_pool():") >= 3
    assert "marlin_make_workspace" in body.split("with back_into_tag_pool():")[-1]


def test_presplit_puffer_wird_im_pool_geboren():
    import inspect

    from sglang.srt.layers.moe import expert_offload as eo

    src = inspect.getsource(eo.presplit_expert_offload_after_repack)
    i = src.index("with back_into_tag_pool():")
    assert "buf = torch.empty(" in src[i : i + 200]


def test_ct_stream_presplit_laeuft_ausserhalb():
    import inspect

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    src = inspect.getsource(FusedMoE._ct_stream_presplit_now)
    i = src.index('outside_tag_pool(reason="ct-stream-presplit")')
    assert "process_weights_after_loading(self)" in src[i : i + 300]


def _block_nach(src, kopf):
    """Die eingerueckten Zeilen des with-Blocks, der mit `kopf` beginnt."""
    zeilen = src.split("\n")
    k = next(n for n, z in enumerate(zeilen) if kopf in z)
    tiefe = len(zeilen[k]) - len(zeilen[k].lstrip())
    block = []
    for z in zeilen[k + 1 :]:
        if z.strip() and len(z) - len(z.lstrip()) <= tiefe:
            break
        block.append(z)
    return "\n".join(block)


@pytest.mark.parametrize("wo", ["ct-stream", "schema"])
def test_empty_cache_laeuft_draussen_nicht_drinnen(wo):
    """fnFL2x3: der Allocator gibt den Default-Pool nur frei, solange KEINE
    Pool-Umleitung aktiv ist. Metall-Probe (echte tag_pool_scope/outside/
    back_into auf der 3080): draussen 1000 -> 400 MiB, drinnen bleibt 900."""
    import inspect

    if wo == "ct-stream":
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        src = inspect.getsource(FusedMoE._ct_stream_presplit_now)
        kopf = 'outside_tag_pool(reason="ct-stream-presplit")'
    else:
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_wNa16_moe as moe,
        )

        src = inspect.getsource(moe.CompressedTensorsWNA16MoE.process_weights_after_loading)
        kopf = 'outside_tag_pool(reason="ct-moe-repack")'
    assert "torch.cuda.empty_cache()" in _block_nach(src, kopf)
