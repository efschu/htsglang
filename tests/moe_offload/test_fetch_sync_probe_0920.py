"""Task #49 probe switch (20.09.): SGLANG_MOE_OFFLOAD_FETCH_SYNC host-synchronizes
after every joined fetch; default off."""

import inspect

from sglang.srt.layers.moe import expert_offload as eo


def test_switch_parses_and_defaults_off(monkeypatch):
    for raw, want in (("", False), ("0", False), ("1", True), ("on", True)):
        eo._FETCH_SYNC["on"] = None
        if raw == "":
            monkeypatch.delenv("SGLANG_MOE_OFFLOAD_FETCH_SYNC", raising=False)
        else:
            monkeypatch.setenv("SGLANG_MOE_OFFLOAD_FETCH_SYNC", raw)
        assert eo.fetch_sync_on() is want, raw
    eo._FETCH_SYNC["on"] = None


def test_the_synchronize_sits_after_the_join():
    src = inspect.getsource(eo.MoEExpertOffloadCache._fetch)
    i = src.index("torch.cuda.current_stream().wait_stream(self._stream)")
    j = src.index("if fetch_sync_on():")
    assert i < j and "torch.cuda.synchronize()" in src[j:]
