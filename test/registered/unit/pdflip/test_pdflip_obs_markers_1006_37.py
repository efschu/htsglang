"""nf-next-1006-37: two observation-only log markers in mamba_component (no behaviour change).

(A) ``#59 FINISH-TRACK rid= cache_len= branching= tokens= extra_buffer=`` at the
    finish INSERT (where ``cache_len`` is taken). nf-next-1006-30 reads
    ``mamba_last_track_seqlen`` after the insert, where the non-extra-buffer path has
    already reset it to None. Every rank logs its own line, finished requests only.
(B) ``MATCH-CENSUS-DEEP rid= reached= accepted= branching= prefix=`` for a HIT of
    depth >= 1000 on attention-TP rank 0 only (the ``[#904 match-census]`` hit line is
    sampled every Nth walk; nf-next-1006-29 could not see the class-B hit).

RED on 0d4768f4f9: neither marker exists.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import logging  # noqa: E402
import re  # noqa: E402
import types  # noqa: E402
from unittest import mock  # noqa: E402

import torch  # noqa: E402

from flliper.srt.mem_cache.base_prefix_cache import MatchResult  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components import mamba_component as mc  # noqa: E402

LOGGER = mc.logger.name
CENSUS = "flliper.srt.mem_cache.match_refusal_census.census_every"


# ---------------------------------------------------------------- (A) FINISH-TRACK
def _finish_self(extra_buffer):
    pool = types.SimpleNamespace(
        replayssm_write_pos=None,
    )
    r2t = types.SimpleNamespace(
        mamba_pool=pool, get_mamba_ping_pong_keep_idx=lambda req: 1
    )
    return types.SimpleNamespace(
        enable_mamba_extra_buffer=extra_buffer,
        mamba_checkpoint_interval=64,
        int8_ckpt_pool=None,
        cache=types.SimpleNamespace(req_to_token_pool=r2t),
        _off_grid_retention_declines=0,
        _protected_retention_declines=0,
        _991_absent_active_slot=0,
        _pdflip_anchor_step_declines=lambda req, cache_len: True,  # unfinished step: decline early
    )


def _finish_req(last_track, branching):
    return types.SimpleNamespace(
        rid="r-37", mamba_pool_idx=torch.tensor([7]), mamba_last_track_seqlen=last_track,
        mamba_branching_seqlen=branching, cache_protected_len=0,
        mamba_ping_pong_track_buffer=torch.tensor([5, 6]),
    )


def _finish(extra_buffer, last_track, branching, tokens, is_finished=True):
    ip = types.SimpleNamespace(mamba_value=None)
    req = _finish_req(last_track, branching)
    ret = mc.MambaComponent._prepare_for_caching_req_impl(
        _finish_self(extra_buffer), req, ip, tokens, is_finished
    )
    return ret, ip


def _lines(caplog, marker):
    return [r.getMessage() for r in caplog.records if marker in r.getMessage()]


def test_finish_track_logs_the_value_at_the_insert_extra_buffer(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    ret, _ = _finish(True, last_track=23232, branching=23232, tokens=24500)
    assert ret == 23232  # behaviour unchanged: the track is the cut
    (line,) = _lines(caplog, "#59 FINISH-TRACK")
    assert re.fullmatch(
        r"#59 FINISH-TRACK rid=r-37 cache_len=23232 branching=23232 tokens=24500 extra_buffer=1",
        line,
    ), line


def test_finish_track_logs_without_extra_buffer_and_no_track(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    ret, _ = _finish(False, last_track=None, branching=None, tokens=23296)
    assert ret == 23296
    (line,) = _lines(caplog, "#59 FINISH-TRACK")
    assert re.fullmatch(
        r"#59 FINISH-TRACK rid=r-37 cache_len=23296 branching=- tokens=23296 extra_buffer=0", line
    ), line


def test_finish_track_names_a_missing_track_as_dash(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    ret, ip = _finish(True, last_track=None, branching=64, tokens=900)
    assert ip.mamba_value is None  # unchanged decline path
    (line,) = _lines(caplog, "#59 FINISH-TRACK")
    assert "cache_len=- branching=64 tokens=900 extra_buffer=1" in line, line


def test_finish_track_only_for_finished_requests(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    _finish(True, last_track=23232, branching=None, tokens=24000, is_finished=False)
    assert not _lines(caplog, "#59 FINISH-TRACK")


# ------------------------------------------------------------- (B) MATCH-CENSUS-DEEP
def _node():
    cd = types.SimpleNamespace(value=torch.tensor([3]), host_value=None)
    return types.SimpleNamespace(component_data=[None, None, cd])


def _match(rid, chunks, best, cow_mamba=False, req=None):
    self_ = types.SimpleNamespace(
        cache=types.SimpleNamespace(cache_controller=None),
        mamba_checkpoint_interval=None, mamba_ckpt_strict_resume=False, component_type=2,
    )
    node = _node()
    n = sum(chunks)
    result = MatchResult(
        device_indices=torch.arange(sum(chunks[:best]), dtype=torch.int64),
        last_device_node=node, last_host_node=node, best_match_node=node,
    )
    req = req or types.SimpleNamespace(rid=rid, mamba_pool_idx=None)
    params = types.SimpleNamespace(cow_mamba=cow_mamba, req=req)
    vc = [torch.zeros(c, dtype=torch.int64) for c in chunks]
    with mock.patch.object(mc, "get_server_args",
                           lambda: types.SimpleNamespace(mamba_cache_chunk_size=64)):
        out = mc.MambaComponent.finalize_match_result(
            self_, result=result, params=params, value_chunks=vc, best_value_len=best
        )
    assert n == sum(len(v) for v in vc)
    return out, req


def _deep(tp0=True, every=64):
    return (mock.patch.object(mc, "_obs_is_tp0", lambda: tp0), mock.patch(CENSUS, lambda: every))


def test_deep_hit_logs_one_line_on_tp0(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    a, b = _deep()
    with a, b:
        out, _ = _match("r-37", [12608, 256], best=1)
    assert out.mamba_branching_seqlen == 12864  # behaviour unchanged
    (line,) = _lines(caplog, "MATCH-CENSUS-DEEP")
    assert re.fullmatch(
        r"MATCH-CENSUS-DEEP rid=r-37 reached=12864 accepted=12608 branching=12864 prefix=12608",
        line,
    ), line


def test_deep_hit_without_branching_names_dash(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    a, b = _deep()
    with a, b:
        _match("r-37", [5000], best=1)
    (line,) = _lines(caplog, "MATCH-CENSUS-DEEP")
    assert "reached=5000 accepted=5000 branching=- prefix=5000" in line, line


def test_shallow_hit_and_non_hit_are_silent(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    a, b = _deep()
    with a, b:
        _match("r-37", [512, 256], best=1)       # hit at 512 < 1000
        _match("r-37b", [999], best=1)           # boundary - 1
        _match("r-37c", [23232], best=0)         # accepted 0: not a hit
    assert not _lines(caplog, "MATCH-CENSUS-DEEP")


def test_depth_boundary_1000_logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    a, b = _deep()
    with a, b:
        _match("r-37", [1000], best=1)
    assert len(_lines(caplog, "MATCH-CENSUS-DEEP")) == 1


def test_only_tp0_logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    a, b = _deep(tp0=False)
    with a, b:
        _match("r-37", [12608, 256], best=1)
    assert not _lines(caplog, "MATCH-CENSUS-DEEP")


def test_disarmed_census_means_silent(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    a, b = _deep(every=0)
    with a, b:
        _match("r-37", [12608, 256], best=1)
    assert not _lines(caplog, "MATCH-CENSUS-DEEP")


def test_volume_per_request_is_bounded(caplog):
    """A parked request re-matches every pass: identical re-match = 0 new lines,
    changed matches are capped at _DEEP_CENSUS_PER_REQ."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    a, b = _deep()
    req = types.SimpleNamespace(rid="r-37", mamba_pool_idx=None)
    with a, b:
        for _ in range(50):
            _match("r-37", [12608, 256], best=1, req=req)
        assert len(_lines(caplog, "MATCH-CENSUS-DEEP")) == 1
        for k in range(1, 20):
            _match("r-37", [12608 + 64 * k, 256], best=1, req=req)
    assert len(_lines(caplog, "MATCH-CENSUS-DEEP")) == mc._DEEP_CENSUS_PER_REQ


def test_no_req_no_line(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    a, b = _deep()
    with a, b:
        mc._obs_match_census_deep(None, 5000, 5000, None, 5000)
    assert not _lines(caplog, "MATCH-CENSUS-DEEP")
