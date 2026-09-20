"""xsn409 (20.09.): the kv_cache resume verdict of a wake is the whole TP
group's -- one refused rank must pull the resumed ranks back to the dormant
image, or the group splits and hangs in its next collective."""

import types

from sglang.srt.managers.scheduler_components import weight_updater as wu

CLS = wu.SchedulerWeightUpdaterManager
VERDICT = CLS._weg2_kv_group_verdict


class _Adapter:
    def __init__(self):
        self.paused = []

    def pause(self, tag):
        self.paused.append(tag)


def _rank(mine_resumed_epoch=7):
    return types.SimpleNamespace(
        tp_cpu_group=object(),
        memory_saver_adapter=_Adapter(),
        scheduler=types.SimpleNamespace(weg2_dormant=False),
        _weg2_kv_resumed_epoch=mine_resumed_epoch,
        _weg2_kv_deferred=False,
    )


def _wire(monkeypatch, votes):
    monkeypatch.setattr(wu.torch.distributed, "get_world_size", lambda group=None: len(votes), raising=False)

    def gather(out, mine, group=None):
        out[:] = list(votes)

    monkeypatch.setattr(wu.torch.distributed, "all_gather_object", gather, raising=False)


def test_all_ok_keeps_the_rank_resumed(monkeypatch):
    _wire(monkeypatch, [True, True, True])
    r = _rank()
    assert VERDICT(r, True, 7) is True
    assert r.memory_saver_adapter.paused == [] and r.scheduler.weg2_dormant is False
    assert r._weg2_kv_resumed_epoch == 7 and r._weg2_kv_deferred is False


def test_one_refused_rank_pulls_a_resumed_rank_back_to_dormant(monkeypatch):
    _wire(monkeypatch, [True, False, True])
    r = _rank()
    assert VERDICT(r, True, 7) is False
    assert r.memory_saver_adapter.paused == [wu.GPU_MEMORY_TYPE_KV_CACHE]
    assert r.scheduler.weg2_dormant is True
    assert r._weg2_kv_resumed_epoch is None and r._weg2_kv_deferred is True


def test_the_refused_rank_itself_pauses_nothing_and_stays_deferred(monkeypatch):
    _wire(monkeypatch, [True, False, True])
    r = _rank(mine_resumed_epoch=None)
    assert VERDICT(r, False, 7) is False
    assert r.memory_saver_adapter.paused == [] and r._weg2_kv_deferred is True


def test_world_size_one_is_the_own_verdict(monkeypatch):
    monkeypatch.setattr(wu.torch.distributed, "get_world_size", lambda group=None: 1, raising=False)
    r = _rank()
    assert VERDICT(r, True, 7) is True and VERDICT(r, False, 7) is False
    assert r.memory_saver_adapter.paused == []


def test_the_late_site_asks_the_group_before_marking_the_epoch_done():
    import inspect

    src = inspect.getsource(CLS)
    i_v = src.index("_weg2_kv_ok = self._weg2_kv_group_verdict(bool(_weg2_kv_ok), _kv_epoch)")
    i_done = src.index("self._weg2_kv_epoch_done = _kv_epoch", i_v)
    assert i_v < i_done
