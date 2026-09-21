"""Task #47 Scheibe 6a (20.09.): the front's --weights-resident form -- both
groups keep their weights mapped, the flip moves only kv_cache, the gathered
weights legs are skipped and the pause order is the empty family."""

import inspect

from sglang.srt.weg2 import front as fr


def _front(**kw):
    return fr.Front("http://127.0.0.1:1", "http://127.0.0.1:2", "D", "t", "/tmp", 1, 2, {}, 1.0, **kw)


def test_resident_form_has_an_empty_weights_family():
    f = _front(weights_resident=True, weight_chunks=8)
    assert f.weights_resident is True and f.weights_tags == []
    g = _front(weight_chunks=8)
    assert g.weights_resident is False and len(g.weights_tags) >= 9  # 8 chunks + base


def test_the_flip_driver_skips_the_legs_and_the_order_under_resident():
    src = inspect.getsource(fr.Front.flip)
    assert "if self.weights_resident:" in src
    assert "WEG2-FLIP-LEGS SKIPPED" in src
    i_skip = src.index("WEG2-FLIP-LEGS SKIPPED")
    i_legs = src.index("/release_memory_occupation\",\n", i_skip)  # the family leg comes after, in the else branch
    assert i_skip < i_legs
    assert 'pause_order, why = [], "weights resident on both groups' in src


def test_the_cli_and_the_launcher_carry_the_switch():
    src = inspect.getsource(fr)
    assert 'ap.add_argument("--weights-resident", action="store_true"' in src
    assert "weights_resident=args.weights_resident," in src
    from sglang.srt.weg2 import launcher as lc

    lsrc = inspect.getsource(lc)
    assert 'ap.add_argument("--flip-weights", choices=("family", "resident"), default="family"' in lsrc
    assert '(["--weights-resident"] if getattr(ns, "flip_weights", "family") == "resident" else [])' in lsrc


def test_resident_env_is_set_and_the_server_side_reads_it(monkeypatch, tmp_path):
    from sglang.srt.weg2 import launcher as lc
    from sglang.srt.managers import weg2_memory_saver as ms

    kw = dict(chunk_layers=0, chunk_count=0, tms_so="", transport="bar1", ring=None)
    env = lc.build_env(str(tmp_path), "venv", "0,1,2", str(tmp_path), False, "t", group="D",
                       flip_weights="resident", **kw)
    assert env[ms.WEIGHTS_RESIDENT_ENV] == "1"
    env27 = lc.build_env(str(tmp_path), "venv", "0,1,2", str(tmp_path), False, "t", group="D", **kw)
    assert ms.WEIGHTS_RESIDENT_ENV not in env27
    monkeypatch.setenv(ms.WEIGHTS_RESIDENT_ENV, "1")
    # the backup-OFF wake lock does not apply: the weights never sleep
    ms.assert_backup_off_wake_refill_is_defined(quantization="compressed-tensors", context="t")
    monkeypatch.setenv(ms.WEIGHTS_RESIDENT_ENV, "0")
    import pytest as _pt

    with _pt.raises(ms.Weg2WakeRefused):
        ms.assert_backup_off_wake_refill_is_defined(quantization="compressed-tensors", context="t")


def test_the_release_handler_refuses_a_weights_tag_under_resident():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu.SchedulerWeightUpdaterManager.release_memory_occupation)
    assert "SGLANG_WEG2_WEIGHTS_RESIDENT=1) but the release named" in src
    assert "_foreign = [t for t in tags if is_weights_family_tag(t)]" in src
    from sglang.srt.weg2 import launcher as lc

    assert '"flip_weights": getattr(ns, "flip_weights", "family")' in inspect.getsource(lc._env_knobs)


def test_w19_dormant_residue_is_not_graded_on_the_resident_arm():
    """fnFL2 v21 (21.09.): D's first sleep measured 11534 MiB on a 3080 worker
    against a pausable-form reserve of 1986 -> W19 STOP, though the residue
    is the resident weight set by design."""
    from sglang.srt.weg2.front import dormant_residue_over

    dc = {"gpu-a": 11534, "gpu-b": 1500}
    reserve = {"gpu-a": 1986, "gpu-b": 1986}
    assert dormant_residue_over(dc, reserve, weights_resident=True) == {}
    assert dormant_residue_over(dc, reserve, weights_resident=False) == {"gpu-a": (11534, 1986)}
    assert dormant_residue_over(dc, {}, weights_resident=False) == {}


def test_quiesce_waits_for_inflight_health_probes():
    """fnFL2 v22 (21.09.): a /health_generate forwarded just before the flip
    was still running on D at the release -> assert not idle -> W29."""
    import asyncio

    from sglang.srt.weg2 import front as fr

    inflight = {"D": 0, "P": 0}
    assert asyncio.run(fr.health_probes_drained(inflight, "D", 1.0)) is None

    inflight = {"D": 2, "P": 0}
    ticks = []

    async def fake_sleep(dt):
        ticks.append(dt)
        if len(ticks) == 3:
            inflight["D"] = 0

    waited, seen = asyncio.run(fr.health_probes_drained(inflight, "D", 5.0, sleep=fake_sleep))
    assert seen == 2 and len(ticks) == 3 and waited < 1.0

    inflight = {"D": 1}

    async def never(dt):
        pass

    waited, seen = asyncio.run(fr.health_probes_drained(inflight, "D", 0.05, sleep=never))
    assert seen == 1 and waited >= 0.05
    import inspect

    src = inspect.getsource(fr.Front.handle_health_generate)
    assert "self._health_inflight[g.name] = self._health_inflight.get(g.name, 0) + 1" in src
    assert "await health_probes_drained(self._health_inflight, g.name, HEALTH_DRAIN_BOUND_S)" in inspect.getsource(fr.Front.quiesce)
