"""fnFA16 (20.09. 14:57Z): the solo draft host must not min-reduce its
capture ladder over the TP group -- the shadows never build a draft graph
runner, so the #631 all_reduce had no second participant and wedged the
host until the workers' sampler-warmup barrier timed out."""

import types

import pytest
import torch.distributed as dist

from sglang.srt.model_executor.runner import base_cuda_graph_runner as bcr


class _Reduced(Exception):
    pass


def _runner(is_solo):
    sa = types.SimpleNamespace(
        cuda_graph_config=types.SimpleNamespace(decode=types.SimpleNamespace(bs=[1, 2, 4])),
        enable_two_batch_overlap=False,
        torch_compile_max_bs=0,
    )
    return types.SimpleNamespace(
        server_args=sa,
        req_to_token_pool=types.SimpleNamespace(size=8),
        is_draft_solo_host=is_solo,
    )


@pytest.fixture
def three_ranks(monkeypatch):
    par = types.SimpleNamespace(tp_size=3, attn_tp_size=1, attn_cp_size=1)
    monkeypatch.setattr(bcr, "get_parallel", lambda: par)
    monkeypatch.setattr(bcr, "require_gathered_buffer", lambda sa: False)
    monkeypatch.setattr(
        bcr, "get_flags",
        lambda: types.SimpleNamespace(capture=types.SimpleNamespace(enable_torch_compile=False)),
    )

    def _boom(*a, **k):
        raise _Reduced()

    monkeypatch.setattr(dist, "all_reduce", _boom)
    yield


def test_solo_draft_host_keeps_its_ladder_local(three_ranks):
    capture_bs, compile_bs = bcr.get_batch_sizes_to_capture(_runner(True))
    assert capture_bs == [1, 2, 4] and compile_bs == []


def test_a_group_member_still_agrees_with_its_peers(three_ranks, monkeypatch):
    from sglang.srt import distributed as d

    monkeypatch.setattr(
        d, "get_tp_group", lambda: types.SimpleNamespace(cpu_group=object()), raising=False
    )
    with pytest.raises(_Reduced):
        bcr.get_batch_sizes_to_capture(_runner(False))
