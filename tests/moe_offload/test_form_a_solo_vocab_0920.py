"""fnFA5 (20.09. 12:24Z): under Form A the solo draft's vocab init must not
gather. The host's embed_tokens/lm_head are already full (tp_size=1, F13),
the workers hold a HostOnlyModule -- the old solo path died on the workers
(HostOnlyModule 'weight') and on the host (Minachist's INT8 embedding has
no `.weight`; the split path shares the modules for that case)."""

import types

import pytest

from sglang.srt import rank_role
from sglang.srt.speculative import eagle_worker_v2 as ew


class _Refuse:
    def __getattr__(self, item):
        raise AssertionError(f"solo init must not touch the target under Form A ({item})")


def _worker(is_host):
    w = ew.EagleDraftWorker.__new__(ew.EagleDraftWorker)
    w._spec_solo_active = True
    w._spec_solo_is_host = is_host
    w.target_worker = _Refuse()
    calls = []
    w.init_lm_head = lambda: calls.append("split")
    return w, calls


def test_host_takes_the_split_path_without_a_gather(monkeypatch):
    monkeypatch.setattr(rank_role, "_INSTALLED_PLAN", object())
    monkeypatch.setattr(
        ew, "_solo_gather_full_vocab_rows",
        lambda *a, **k: pytest.fail("gather issued under Form A"),
    )
    w, calls = _worker(is_host=True)
    assert ew.EagleDraftWorker._solo_init_lm_head(w) is None
    assert calls == ["split"]


def test_worker_does_nothing(monkeypatch):
    monkeypatch.setattr(rank_role, "_INSTALLED_PLAN", object())
    w, calls = _worker(is_host=False)
    assert ew.EagleDraftWorker._solo_init_lm_head(w) is None
    assert calls == []


def test_classic_solo_still_reaches_the_target(monkeypatch):
    monkeypatch.setattr(rank_role, "_INSTALLED_PLAN", None)
    w, calls = _worker(is_host=True)
    with pytest.raises(AssertionError, match="must not touch"):
        ew.EagleDraftWorker._solo_init_lm_head(w)
    assert calls == []
