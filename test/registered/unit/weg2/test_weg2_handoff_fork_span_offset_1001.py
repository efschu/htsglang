"""Hand-off store read: the fork span reaches P's end anchor (R1), and the
fallback key list starts at the span's absolute start (R2).

Measured on N3c (boot 10012013, fork anchor on):

R1  P keyed its fork-cut leg in the EXACT bigram form -- F units for F
    committed tokens, the held-back fork token as the last unit's partner
    (``END-ANCHOR units=40132/40132``, ``HANDOFF page_keys=40132``) -- and its
    end anchor sits on that last unit. D's span ended at the fork token F, and
    a bigram key of F tokens has F-1 units: every hand-off read asked one key
    short of its own end anchor (``FETCH CAP keys=40131``, anchors up to the
    previous grid anchor 36863) and D recomputed 2585-4053 tokens. Siblings
    with a longer prompt found the same anchor at once (7-26 read 6-22's).

R2  The tree's fallback list ``operation.weg2_page_keys`` was sliced at
    ``len(ids) - span``. P hands over all N ids of a request it trimmed, so the
    offset came out 6 too deep (``offset=36870`` for a span starting at 36864):
    D read P's keys 6 positions ahead and placed their KV and the end anchor's
    state 6 tokens early (``LOADBACK prefix moved to 40126`` of a 40132 anchor).
"""

from array import array
from pathlib import Path
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import bigram_anchor_key  # noqa: E402
from sglang.srt.weg2 import fork_anchor  # noqa: E402
from sglang.srt.weg2.handoff_keys import keys_for_span, span_key_offset  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TOKEN_ENV = "SGLANG_WEG2_FORK_ANCHOR_TOKEN"
IM = 248045                                   # <|im_start|> on Qwen3.8-27B
GEN = [IM, 74455, 198, 248068, 198]           # "<|im_start|>assistant\n<think>\n"
BODY = list(range(2000, 2018))
PROMPT = BODY + GEN                           # N = 23
N = len(PROMPT)
F = len(BODY)                                 # the fork cut: 18 = N-5
RID = "weg2-5-5"
ROOT = Path(__file__).resolve().parents[4]


def _dreq(ids=PROMPT, **kw):
    base = dict(rid=RID, origin_input_ids=array("q", ids), output_ids=[], return_logprob=False,
                input_embeds=None, session_id=None, multimodal_inputs=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _p_chain(ids, fork):
    """P's exact-bigram key of its fork-cut leg (the units its end anchor is
    filed under), exactly as `_weg2_note_end_anchor` builds it."""
    return bigram_anchor_key(array("q", ids), fork, None, is_bigram=True, exact=True, page_size=1)


def _units(key):
    """The bigram units of ``key`` (pair i = tokens i, i+1); ``len(key)`` of them."""
    t = list(key.token_ids)
    return [(t[i], t[i + 1]) for i in range(len(key))]


def _d_span_key(ids, matched, match_end):
    """D's store-read key of `ids[matched:match_end]` (prefetch_from_storage)."""
    return RadixKey(array("q", ids[matched:match_end]), None, is_bigram=True).page_aligned(1)


# -- R1: the fork span covers P's end-anchor unit ------------------------------


def test_fork_span_includes_the_held_back_token_under_exact_bigram(monkeypatch):
    from sglang.srt.managers import scheduler as S

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv(TOKEN_ENV, str(IM))
    assert fork_anchor.fork_cut(PROMPT, IM) == F
    assert S._weg2_fork_match_end(_dreq(), N - 1, exact_bigram=True) == F + 1
    # upstream keying (no exact form): P's leg has F-1 units -- the span stays F
    assert S._weg2_fork_match_end(_dreq(), N - 1) == F
    assert S._weg2_fork_match_end(_dreq(), N - 1, exact_bigram=False) == F
    # never above the caller's end
    assert S._weg2_fork_match_end(_dreq(), F + 1, exact_bigram=True) == F + 1
    assert S._weg2_fork_match_end(_dreq(), F - 3, exact_bigram=True) == F - 3
    # D still has the generation prompt to extend
    assert F + 1 <= N - 1


def test_d_read_reaches_p_end_anchor_unit(monkeypatch):
    """The read asks exactly P's units: the last one carries the end anchor,
    so D resumes at F and extends only the generation prompt (lost <= 64)."""
    from sglang.srt.managers import scheduler as S

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv(TOKEN_ENV, str(IM))
    p_key = _p_chain(PROMPT, F)
    assert len(p_key) == F, "P's exact form: F units for F committed tokens"
    end = S._weg2_fork_match_end(_dreq(), N - 1, exact_bigram=True)
    d_key = _d_span_key(PROMPT, 0, end)
    assert len(d_key) == len(p_key), (len(d_key), len(p_key))
    assert _units(d_key) == _units(p_key)
    assert _units(p_key)[-1] == (BODY[-1], IM), "the end-anchor unit pairs the fork token"
    # the hit query clips P's chain to the span's own units
    # (`_k = min(len(_hk), len(own))`): the end-anchor unit survives the clip
    k = min(len(p_key), len(d_key))
    assert k == F
    lost = N - k
    assert lost <= 64 and lost == len(GEN)


def test_tail_read_after_a_grid_anchor_also_reaches_it(monkeypatch):
    from sglang.srt.managers import scheduler as S

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv(TOKEN_ENV, str(IM))
    matched = 8
    end = S._weg2_fork_match_end(_dreq(), N - 1, exact_bigram=True)
    p_key = _p_chain(PROMPT, F)
    span = keys_for_span([str(u) for u in _units(p_key)], matched, end - matched, 1)
    d_key = _d_span_key(PROMPT, matched, end)
    assert len(d_key) == F - matched
    assert [str(u) for u in _units(d_key)] == list(span[: len(d_key)])
    assert matched + min(len(span), len(d_key)) == F


def test_wiring_scheduler_passes_the_exact_bigram_form():
    src = (ROOT / "python/sglang/srt/managers/scheduler.py").read_text()
    a = src.index("_match_end = req._compute_max_prefix_len(")
    b = src.index("_match_end = _weg2_fork_match_end(")
    c = src.index("_new_input_tokens = req.full_untruncated_fill_ids[_matched_len:_match_end]")
    assert a < b < c
    call = src[b:c]
    assert 'exact_bigram=bool(getattr(self.tree_cache, "is_eagle", False))' in call
    assert 'bool(getattr(self.tree_cache, "bigram_anchor_exact", False))' in call


# -- R2: the fallback key list starts at the span's absolute start ------------


def test_offset_is_the_span_start_for_a_trimmed_hand_off():
    # N3c weg2-5-5: N=40137 ids handed over, the tail read starts at 36864 and
    # asks 3267 (bigram) units -- the legacy form gave 36870
    assert span_key_offset(36864, 40137, 3267, 1) == 36864
    assert span_key_offset(None, 40137, 3267, 1) == 40137 - 3267, "legacy without a base"
    # from-root read of an N-1 trimmed request (N2: legacy offset 2)
    assert span_key_offset(0, 4061, 4059, 1) == 0
    # host-pool truncated span: still the start
    assert span_key_offset(1024, 90000, 512, 1) == 1024
    # paged
    assert span_key_offset(128, 1000, 100, 64) == 2


def test_offset_agrees_with_the_registry_slice():
    """The fallback list must be the registry's list (keys_for_span)."""
    chain = [f"k{i}" for i in range(40132)]
    matched, n_tok = 36864, 3268
    reg = keys_for_span(chain, matched, n_tok, 1)
    off = span_key_offset(matched, 40137, 3267, 1)
    fallback = chain[off:off + 3267]
    assert fallback == reg[:3267]


def test_wiring_tree_uses_the_helper_and_scheduler_passes_key_base():
    tree = (ROOT / "python/sglang/srt/mem_cache/unified_radix_cache.py").read_text()
    assert "_off = _span_key_offset(key_base, _ntok, int(prefetch_length), int(self.page_size))" in tree
    assert "(int(_ntok) - int(prefetch_length)) // int(self.page_size)" not in tree
    sch = (ROOT / "python/sglang/srt/managers/scheduler.py").read_text()
    assert sch.count("key_base=int(_matched_len),") == 2, "both prefetch_from_storage calls"
