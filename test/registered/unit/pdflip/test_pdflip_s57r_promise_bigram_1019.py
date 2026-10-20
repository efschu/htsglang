"""S57r (review 19.10.2026, S57 Nebenbefund 2): p_twin_defer's END / chunk promise is
bounded by the twin's claim B = floor_page(shared - 1), not by shared.

A twin sharing ids[:shared] claims at most B (bigram keys take the next token too --
twin_anchor.boundary, the bound every other promise kind already used). With
shared % page == 0 an END anchor or chunk end AT shared lies one token past that claim:
rc15 rule 5 promised it anyway (PROMISE_END / PROMISE_CHUNK), the twin waited for the
source and then resumed at a shallower anchor. Danger direction of the fix: none of
the kept promises may vanish (shared % page != 0 unchanged), and a dropped END
promise falls through to the boundary rules (the twin stays held for B itself).
"""

from types import SimpleNamespace

import pytest

from flliper.srt.pdflip import p_twin_defer as d
from flliper.srt.pdflip import twin_anchor as twa
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

PAGE = 64


@pytest.fixture(autouse=True)
def _clean():
    twa._reset_for_test()
    yield
    twa._reset_for_test()


def _st(chunk=16384, s0=None, rid="src"):
    st = d._State(min_tokens=1, wait_s=10.0, settle_s=0.1, settle_passes=1,
                  page=PAGE, chunk=chunk, promise=True)
    if s0 is not None:
        st.s0[rid] = int(s0)
    return st


def _src(n, rid="src"):
    return SimpleNamespace(rid=rid, origin_input_ids=list(range(1, n + 1)))


def test_end_at_shared_on_a_page_edge_is_no_end_promise():
    # end = floor_page(n - 1) = 4096 = shared; B = 4032
    src = _src(4100)
    st = _st(s0=1024)
    kind, pos = d._promise(st, src, 4096)
    assert kind != d.PROMISE_END
    assert pos <= twa.boundary(4096, PAGE) == 4032


def test_end_below_the_claim_is_still_promised():
    src = _src(4100)
    for shared in (4097, 4100, 4159):           # B = 4096 = end
        assert d._promise(_st(s0=1024), src, shared) == (d.PROMISE_END, 4096)


def test_chunk_end_at_shared_on_a_page_edge_is_not_promised_there():
    src = _src(9000)
    st = _st(chunk=2048, s0=0)
    kind, pos = d._promise(st, src, 4096)
    assert (kind, pos) == (d.PROMISE_CHUNK, 2048)      # rc15: (chunk, 4096)
    assert d._promise(_st(chunk=2048, s0=0), src, 4100) == (d.PROMISE_CHUNK, 4096)


def test_rule4_gain_uses_the_same_bound():
    src = _src(4100)
    assert d._gain(_st(s0=1024), src, 4096) is False   # rc15: True (end == shared)
    assert d._gain(_st(s0=1024), src, 4100) is True
    assert d._gain(_st(chunk=2048, s0=1024), src, 4096) is True   # a chunk end fits below B
