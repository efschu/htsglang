"""weg2xsn289/291 (18.09.): D's per-token KV cell charged the DFlash draft
layers (28672 -> 43008 B/token) for a mirror draft pool that the window pool
never allocates (4113 rows); 2.4 GiB idle per 3080, the 5090 sized at 22k
tokens. Under the window pool the cell is the target cell and the draft is a
constant reserve off the pool budget."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.model_executor import pool_configurator as pc  # noqa: E402


def _mr(window=2048, draft_worker=False, dflash=True, mrr=6, block=8):
    return SimpleNamespace(
        is_draft_worker=draft_worker,
        spec_algorithm=SimpleNamespace(is_dflash_family=lambda: dflash),
        server_args=SimpleNamespace(speculative_draft_window_size=window,
                                    speculative_num_draft_tokens=block,
                                    max_running_requests=mrr),
        tp_rank=1,
    )


def test_window_pool_slots_match_the_workers_formula(monkeypatch):
    monkeypatch.delenv("SGLANG_DFLASH_SOLO_POOL_FACTOR", raising=False)
    sa = SimpleNamespace(speculative_num_draft_tokens=8, max_running_requests=1)
    assert pc.window_pool_draft_slots(sa, 2048) == 4113          # xsn289's D log, verbatim
    sa6 = SimpleNamespace(speculative_num_draft_tokens=8, max_running_requests=6)
    assert pc.window_pool_draft_slots(sa6, 2048) == 1 + (2048 + 8) * 6 * 2


def test_under_the_window_pool_the_cell_is_the_target_cell_and_the_draft_a_reserve(monkeypatch):
    monkeypatch.setenv("SGLANG_DFLASH_WINDOW_POOL", "1")
    monkeypatch.delenv("SGLANG_DFLASH_SOLO_POOL_FACTOR", raising=False)
    cell, reserve = pc.apply_window_pool_draft_charge(_mr(mrr=1), 28672, 43008)
    assert cell == 28672 and reserve == 4113 * 14336            # 59 MB, not 2.4 GiB
    # 6.877 GiB on TP1 (xsn289): 171,681 tokens at 43008 -> ~249k at 28672 minus the reserve
    avail = 7383670784
    assert (avail - reserve) // 28672 > 245_000


@pytest.mark.parametrize("why", ["env-off", "draft-worker", "no-window", "not-dflash", "no-draft-part"])
def test_every_other_path_is_byte_identical(monkeypatch, why):
    monkeypatch.setenv("SGLANG_DFLASH_WINDOW_POOL", "1")
    mr = _mr()
    cell_in = 43008
    if why == "env-off":
        monkeypatch.setenv("SGLANG_DFLASH_WINDOW_POOL", "0")
    elif why == "draft-worker":
        mr = _mr(draft_worker=True)
    elif why == "no-window":
        mr = _mr(window=None)
    elif why == "not-dflash":
        mr = _mr(dflash=False)
    elif why == "no-draft-part":
        cell_in = 28672
    assert pc.apply_window_pool_draft_charge(mr, 28672, cell_in) == (cell_in, 0)


def test_calculate_pool_sizes_takes_the_reserve_off_the_budget():
    src = open(pc.__file__).read()
    k = src.index("available_bytes // self._cell_size")
    assert "_window_pool_reserve_bytes" in src[k - 900:k]
    j = src.index("apply_solo_draft_kv_cell_factor(\n            mr, target_cell_size, self._cell_size\n        )")
    assert "apply_window_pool_draft_charge(" in src[j:j + 500]
