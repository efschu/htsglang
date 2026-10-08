"""L3-FORMAT (N2, 01.10.): the persistent store records the BYTE LAYOUT it holds.

Needle-MISS analysis: the persistent L3 key/identity carried the model,
override and weight fingerprint, and the directory the KV dtype -- but nothing
named the layout of the bytes themselves (the canonical KV page, the mamba
blob's temporal/conv order and state dtype). A layout change in code would
have read old bytes as new ones without a word. Now ONE ``L3_FORMAT.json``
per store (every group: P writes what D reads), recorded by the first rank
(link(2), like the rank identity), compared by every rank, a mismatch W165 by
name. Deliberately not an image rev: an image that keeps the layout keeps the
store."""
from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402


def _fake(root, fmt):
    from flliper.srt.mem_cache.hicache_storage import HiCacheFile

    fake = mock.MagicMock()
    fake.file_path = root
    fake._l3p_on = HiCacheFile._l3p_on
    fake._l3p_persistent_dir = lambda: HiCacheFile._l3p_persistent_dir(fake)
    cfg = mock.MagicMock()
    cfg.l3_rank_format = fmt
    return HiCacheFile._l3p_check_rank_format, fake, cfg


def _sa(model, **kw):
    base = dict(model_path=model, kv_cache_dtype="fp8_e4m3", page_size=1, mamba_ssm_dtype="bfloat16",
                dtype="auto", enable_linear_replayssm=False, enable_linear_replayssm_spec=False,
                rank_tp_ratio=None)
    base.update(kw)
    return mock.MagicMock(**base)


def _store(root):
    with open(os.path.join(root, "L3_IDENTITY.json"), "w") as f:
        f.write("{}")


def test_format_recorded_then_matched_then_a_layout_bump_refused(monkeypatch):
    from flliper.srt.mem_cache import hicache_storage as hs
    from flliper.srt.mem_cache.pdflip_store_gates import PdFlipL3IdentityMismatch

    monkeypatch.setenv("FLLIPER_PDFLIP_L3_PERSIST", "1")
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as m:
        _store(root)
        fmt = hs.l3_rank_format(_sa(m))
        fn, fake, cfg = _fake(root, fmt)
        fn(fake, cfg)                                   # a legacy store gets its record
        with open(os.path.join(root, "L3_FORMAT.json")) as f:
            assert json.load(f) == fmt
        fn(fake, cfg)                                   # match
        monkeypatch.setattr(hs, "L3_FORMAT_GENERATION", hs.L3_FORMAT_GENERATION + 1)
        fn2, fake2, cfg2 = _fake(root, hs.l3_rank_format(_sa(m)))
        with pytest.raises(PdFlipL3IdentityMismatch) as ei:
            fn2(fake2, cfg2)
        assert "W165" in str(ei.value) and "format" in str(ei.value)


def test_a_state_dtype_change_is_refused(monkeypatch):
    from flliper.srt.mem_cache import hicache_storage as hs
    from flliper.srt.mem_cache.pdflip_store_gates import PdFlipL3IdentityMismatch

    monkeypatch.setenv("FLLIPER_PDFLIP_L3_PERSIST", "1")
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as m:
        _store(root)
        fn, fake, cfg = _fake(root, hs.l3_rank_format(_sa(m)))
        fn(fake, cfg)
        fn2, fake2, cfg2 = _fake(root, hs.l3_rank_format(_sa(m, mamba_ssm_dtype="float32")))
        with pytest.raises(PdFlipL3IdentityMismatch):
            fn2(fake2, cfg2)


def test_p_and_d_of_the_27b_form_share_one_record(monkeypatch):
    """The 27B boot's own args (P: no ReplaySSM, D: ReplaySSM-spec; ratios
    differ): one record, no refusal between the groups."""
    from flliper.srt.mem_cache import hicache_storage as hs

    with tempfile.TemporaryDirectory() as m:
        p = hs.l3_rank_format(_sa(m, rank_tp_ratio=None))
        d = hs.l3_rank_format(_sa(m, enable_linear_replayssm_spec=True, rank_tp_ratio="58,25,25"))
        assert p == d
        assert p["format_gen"] == hs.L3_FORMAT_GENERATION


def test_unset_state_dtype_resolves_from_the_model_config(monkeypatch):
    from flliper.srt.mem_cache import hicache_storage as hs

    with tempfile.TemporaryDirectory() as m:
        with open(os.path.join(m, "config.json"), "w") as f:
            json.dump({"text_config": {"mamba_ssm_dtype": "bfloat16"}}, f)
        assert hs.l3_rank_format(_sa(m, mamba_ssm_dtype=None)) == hs.l3_rank_format(_sa(m))


def test_not_a_persistent_dir_records_nothing(monkeypatch):
    from flliper.srt.mem_cache import hicache_storage as hs

    monkeypatch.setenv("FLLIPER_PDFLIP_L3_PERSIST", "1")
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as m:
        fn, fake, cfg = _fake(root, hs.l3_rank_format(_sa(m)))
        fn(fake, cfg)
        assert os.listdir(root) == []


def test_wired_after_the_rank_identity_and_through_the_controller():
    import inspect

    from flliper.srt.managers import cache_controller as cc
    from flliper.srt.mem_cache import hicache_storage as hs

    src = inspect.getsource(hs)
    i = src.index("self._l3p_check_rank_identity(storage_config)")
    assert src.index("self._l3p_check_rank_format(storage_config)") > i
    assert "l3_rank_format=_l3_rank_format_or_none(server_args)" in inspect.getsource(cc)
