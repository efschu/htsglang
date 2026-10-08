"""ZR-4 (01.10.): two requests in one D admission pass do not compute the same
tokens twice.

Metal y6h (D log ...dauer10011531, TP0, 15:42:50): '#969 EXTENT n=51 reqs=[
('pdflip-10-', 42880, 44402, 42880, 1522), ..., ('pdflip-10-', 42880, 44231,
42880, 1351)]' -- two turns of one agent, both matched to 42880, both
extended from there in the same forward.
"""

import inspect
from types import SimpleNamespace

import pytest

from flliper.srt.managers.schedule_batch import Range
from flliper.srt.managers.schedule_policy import PrefillAdder
from flliper.srt.pdflip import d_twin_pass as tw

BASE = list(range(42880))
SHARED = list(range(1_000_000, 1_000_000 + 1200))


def _req(rid, tail, prefix=42880, rng=None):
    r = SimpleNamespace(rid=rid, extra_key=None, full_untruncated_fill_ids=BASE + tail)
    r.prefix_indices = [0] * prefix
    r.extend_range = rng
    return r


def _first(tail_extra=322):
    ids = SHARED + list(range(2_000_000, 2_000_000 + tail_extra))
    return _req("pdflip-10-a", ids, rng=Range(42880, 42880 + len(ids)))


@pytest.fixture
def d(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.delenv(tw.ENV, raising=False)
    monkeypatch.delenv(tw.ENV_MIN_TOKENS, raising=False)


def test_twin_waits_one_pass(d):
    a = _first()
    b = _req("pdflip-10-c", SHARED + list(range(3_000_000, 3_000_151)))
    assert tw.overlap(b, 42880, a) == 1200
    assert tw.waits(b, 42880, [a])


def test_diverging_turns_do_not_wait(d):
    a = _first()
    b = _req("pdflip-10-c", list(range(3_000_000, 3_001_351)))
    assert tw.overlap(b, 42880, a) == 0
    assert not tw.waits(b, 42880, [a])


def test_below_one_page_does_not_wait(d):
    a = _first()
    b = _req("pdflip-10-c", SHARED[:40] + list(range(3_000_000, 3_001_300)))
    assert not tw.waits(b, 42880, [a])


def test_twin_beyond_the_others_chunk_does_not_wait(d):
    a = _first()
    a.extend_range = Range(42880, 42880 + 32)  # this pass computes only 32 tokens
    b = _req("pdflip-10-c", SHARED + [7])
    assert not tw.waits(b, 42880, [a])


def test_p_group_is_untouched(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    a = _first()
    b = _req("pdflip-10-c", SHARED + [7])
    assert not tw.waits(b, 42880, [a])


def test_switch_off(d, monkeypatch):
    monkeypatch.setenv(tw.ENV, "0")
    a = _first()
    b = _req("pdflip-10-c", SHARED + [7])
    assert not tw.waits(b, 42880, [a])


def test_wired_after_the_skip_wait():
    src = inspect.getsource(PrefillAdder.add_one_req)
    gate = src.index("d_twin_pass.waits(")
    assert src.index("tail_adopt.skip_waits(") < gate < src.index("_tail = tail_adopt.plan_adopt(")
    assert "not tail_adopt.skip_joinable(" in src[gate - 200:gate]
