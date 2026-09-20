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
