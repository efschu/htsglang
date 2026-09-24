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


def test_the_verdict_sits_between_the_resume_and_the_clear_half():
    """xsn410: after the clear half (xsn409's placement) the verdict deadlocked --
    the resumed ranks ran the clear half's own collective
    (_weg2_release_dormant_hold -> _weg2_group_min_flags) while the refused
    rank waited in the verdict's all_gather. Now: every exit of the resume
    half votes, and the clear half runs on no rank unless the group resumed."""
    import inspect

    src = inspect.getsource(CLS)
    i_part = src.index("def _weg2_kv_resume_part():")
    i_clear = src.index("def _weg2_kv_clear_part():", i_part)
    part = src[i_part:i_clear]
    # the two refusals vote False; the landed resume votes True before the epoch mark
    assert part.count("return self._weg2_kv_group_verdict(False, _kv_epoch)") == 2
    i_yes = part.index("if not self._weg2_kv_group_verdict(True, _kv_epoch):")
    i_mark = part.index("self._weg2_kv_resumed_epoch = _kv_epoch")
    i_resume = part.index("self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)")
    assert i_resume < i_yes < i_mark
    # the late site no longer votes a second time
    late = src[i_clear:]
    assert "_weg2_kv_group_verdict(bool(_weg2_kv_ok)" not in late


def test_the_plan_and_the_mid_site_are_group_uniform():
    """The resume half now carries a collective, so the decisions that gate it
    (early/late plan, mid-legs resume) must be the GROUP's, not one rank's."""
    import inspect

    src = inspect.getsource(CLS)
    assert 'self._weg2_kv_group_all(_fundable, "WAKE-KV-FIRST fundable")' in src
    assert "fundable=_fundable," in src
    assert '_mid_ok = self._weg2_kv_group_all(bool(_mid_ok), "WAKE-KV-MID tag=%s" % (tag,))' in src


def test_group_all_is_an_and_over_the_votes(monkeypatch):
    ALL = CLS._weg2_kv_group_all
    _wire(monkeypatch, [True, True, True])
    assert ALL(_rank(), True, "x") is True
    _wire(monkeypatch, [True, False, True])
    assert ALL(_rank(), True, "x") is False
    monkeypatch.setattr(wu.torch.distributed, "get_world_size", lambda group=None: 1, raising=False)
    assert ALL(_rank(), True, "x") is True and ALL(_rank(), False, "x") is False
