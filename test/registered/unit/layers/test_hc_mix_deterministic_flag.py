"""fn1s boot 2026-09-16: the hyper-connection mix asked upstream's grouped
exec view for ``deterministic.enable_deterministic_inference``; this line
keeps the flag flat on ServerArgs."""

from types import SimpleNamespace

from sglang.srt.layers import hc_mix_triton as m


def test_reads_the_flat_flag_and_defaults_to_false(monkeypatch):
    import sglang.srt.runtime_context as rc

    monkeypatch.setattr(rc, "get_server_args", lambda: SimpleNamespace(enable_deterministic_inference=True))
    assert m._deterministic_inference() is True
    monkeypatch.setattr(rc, "get_server_args", lambda: SimpleNamespace())
    assert m._deterministic_inference() is False

    def boom():
        raise ValueError("no server args published")

    monkeypatch.setattr(rc, "get_server_args", boom)
    assert m._deterministic_inference() is False
