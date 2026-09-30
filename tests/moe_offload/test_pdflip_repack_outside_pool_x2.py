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

from flliper.srt.managers import pdflip_memory_saver as MS


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
    # Ohne Karte: kein temporaerer Pool (MemPool() scheitert) -- die
    # begin/end-Reihenfolge des Tag-Pools ist davon unabhaengig.
    monkeypatch.setattr(MS, "_transient_pool", _no_pool)
    return calls


from contextlib import contextmanager  # noqa: E402


@contextmanager
def _no_pool(reason):
    yield None


def test_back_into_without_exit_is_a_no_op(stepped):
    with MS.back_into_tag_pool() as inside:
        assert inside is False
    assert stepped == []


def test_survivor_is_born_in_pool(stepped):
    with MS.outside_tag_pool(reason="x"):
        assert stepped == ["end"]
        with MS.back_into_tag_pool() as inside:
            assert inside is True
            assert stepped == ["end", "begin"]
        assert stepped == ["end", "begin", "end"]
    assert stepped == ["end", "begin", "end", "begin"]
    assert MS._STEPPED_OUT_POOLS == []


def test_nested_outside_does_not_touch_allocator(stepped):
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


def test_exception_in_survivor_restores_state(stepped):
    with pytest.raises(RuntimeError):
        with MS.outside_tag_pool():
            with MS.back_into_tag_pool():
                raise RuntimeError("alloc blew up")
    assert stepped == ["end", "begin", "end", "begin"]
    assert MS._STEPPED_OUT_POOLS == []


class _TmpPool:
    made = []

    def __init__(self):
        self.id = (0, 90 + len(_TmpPool.made))
        _TmpPool.made.append(self)


@pytest.fixture
def tmp_pools(monkeypatch):
    """Der temporaere Pool: angelegt, betreten, und je nach Lebendzustand
    geloescht oder behalten."""
    _TmpPool.made.clear()
    entered = []

    @contextmanager
    def use_mem_pool(pool):
        entered.append(pool.id)
        yield

    monkeypatch.setattr(torch.cuda, "MemPool", _TmpPool, raising=False)
    monkeypatch.setattr(torch.cuda, "use_mem_pool", use_mem_pool, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(MS, "_LOAD_TRANSIENT_POOL", None, raising=False)
    monkeypatch.setattr(MS, "_KEPT_TRANSIENT_POOLS", [], raising=False)
    return entered


def test_one_load_pool_for_all_blocks(tmp_pools, monkeypatch):
    """Metall-Probe 22.09.: ein Pool je Block assertet beim Loeschen im
    Saver-Bereich (captures_underway.empty()). Also EIN Pool fuers Laden."""
    monkeypatch.setattr(MS, "_LOAD_TRANSIENT_POOL", None, raising=False)
    for _ in range(3):
        with MS._transient_pool("x") as tmp:
            assert tmp is _TmpPool.made[0]
    assert len(_TmpPool.made) == 1
    assert tmp_pools == [(0, 90)] * 3


def test_release_refused_while_redirect_is_open(tmp_pools, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(MS, "_LOAD_TRANSIENT_POOL", _TmpPool(), raising=False)
    monkeypatch.setattr(MS, "_ACTIVE_TAG_POOL", _FakePool(), raising=False)
    monkeypatch.setattr(MS, "_tms_cdll_in_region", lambda: None)
    assert MS.release_load_transient_pool("x") == 0.0
    assert MS._LOAD_TRANSIENT_POOL is not None          # nicht angefasst
    monkeypatch.setattr(MS, "_ACTIVE_TAG_POOL", None, raising=False)
    monkeypatch.setattr(MS, "_tms_cdll_in_region", lambda: object())
    assert MS.release_load_transient_pool("x") == 0.0   # Saver-Bereich offen
    assert MS._LOAD_TRANSIENT_POOL is not None


def test_release_keeps_pool_with_live_block(tmp_pools, monkeypatch):
    """Loeschen mit lebendem Block = v56 (invalid argument)."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    pool = _TmpPool()
    monkeypatch.setattr(MS, "_LOAD_TRANSIENT_POOL", pool, raising=False)
    monkeypatch.setattr(MS, "_ACTIVE_TAG_POOL", None, raising=False)
    monkeypatch.setattr(MS, "_STEPPED_OUT_POOLS", [], raising=False)
    monkeypatch.setattr(MS, "_tms_cdll_in_region", lambda: None)
    monkeypatch.setattr(MS, "_pool_has_live_blocks", lambda p, reason: True)
    assert MS.release_load_transient_pool("x") == 0.0
    assert MS._KEPT_TRANSIENT_POOLS == [pool]
    assert MS._LOAD_TRANSIENT_POOL is None


def test_model_runner_frees_load_pool_after_region():
    import inspect

    from flliper.srt.model_executor import model_runner as mr

    src = inspect.getsource(mr.ModelRunner.load_model)
    i = src.index('release_load_transient_pool(reason="after-load")')
    row_idx = src[: i].split("\n")[-1]
    # auf Methoden-Ebene (8 Leerzeichen), also NACH dem with weights_region-Block
    assert row_idx == " " * 8, repr(row_idx)


def test_outside_leaves_tag_pool_before_entering_temporary(
    tmp_pools, monkeypatch
):
    """Kein Doppeleintrag desselben Pools: end(tag) VOR use_mem_pool(tmp),
    begin(tag) NACH dessen Austritt."""
    calls = []
    monkeypatch.setattr(MS, "_ACTIVE_TAG_POOL", _FakePool(), raising=False)
    monkeypatch.setattr(MS, "_STEPPED_OUT_POOLS", [], raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(MS, "_pool_has_live_blocks", lambda pool, reason: False)
    mod = types.ModuleType("torch.cuda.memory")
    mod._cuda_endAllocateToPool = lambda dev, pid: calls.append(("end", pid))
    mod._cuda_beginAllocateCurrentThreadToPool = lambda dev, pid: calls.append(
        ("begin", pid)
    )
    monkeypatch.setitem(__import__("sys").modules, "torch.cuda.memory", mod)
    with MS.outside_tag_pool(reason="x"):
        calls.append(("drin", tmp_pools[-1]))
        with MS.back_into_tag_pool():
            calls.append("ueberlebender")
    assert calls == [
        ("end", (0, 7)),
        ("drin", (0, 90)),
        ("begin", (0, 7)),
        "ueberlebender",
        ("end", (0, 7)),
        ("begin", (0, 7)),
    ]


def test_schema_wrapper_exits_and_repack_enters_for_survivors():
    import inspect

    from flliper.srt.layers.quantization.compressed_tensors.schemes import (
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


def test_presplit_buffer_is_born_in_pool():
    import inspect

    from flliper.srt.layers.moe import expert_offload as eo

    src = inspect.getsource(eo.presplit_expert_offload_after_repack)
    i = src.index("with back_into_tag_pool():")
    assert "buf = torch.empty(" in src[i : i + 200]


def test_ct_stream_presplit_runs_outside():
    import inspect

    from flliper.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    src = inspect.getsource(FusedMoE._ct_stream_presplit_now)
    i = src.index('reason="ct-stream-presplit"')
    assert "process_weights_after_loading(self)" in src[i : i + 300]


def _block_nach(src, header):
    """Die eingerueckten Zeilen des with-Blocks, der mit `kopf` beginnt."""
    row_list = src.split("\n")
    k = next(n for n, z in enumerate(row_list) if header in z)
    depth = len(row_list[k]) - len(row_list[k].lstrip())
    block = []
    for z in row_list[k + 1 :]:
        if z.strip() and len(z) - len(z.lstrip()) <= depth:
            break
        block.append(z)
    return "\n".join(block)


def test_ct_stream_presplit_runs_in_layer_chunk():
    """fnFL2x5: der ct-stream-Presplit lief im BASIS-Tag, der Austausch benennt
    jedes Stueck von Layer N nach seinem Chunk -> P's Wake sammelte das
    Experten-Praefix von Layer 29 unter weights_9 in Seiten, die erst mit dem
    Basis-Tag (zuletzt) gemappt wurden -> Segfault in memcpy_async. Der
    Endpass des Laders (loader.py) tut es richtig; derselbe Rahmen hier."""
    import inspect

    from flliper.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    src = inspect.getsource(FusedMoE._ct_stream_presplit_now)
    code = "\n".join(z for z in src.split("\n") if not z.lstrip().startswith("#"))
    w = code.index("with weight_chunk_scope(self.layer_id)")
    o = code.index("outside_tag_pool(", w)
    p = code.index("self.quant_method.process_weights_after_loading(self)", o)
    # alle drei im selben with-Kopf bzw. darin geschachtelt: kein Doppelpunkt-
    # Ende eines Blocks zwischen Scope-Kopf und Repack auf gleicher Tiefe
    head_depth = len(code[: w].split("\n")[-1])
    between = code[w:p].split("\n")[1:]
    assert all(
        len(z) - len(z.lstrip()) >= head_depth for z in between if z.strip()
    ), "the repack is not in chunk scope"
