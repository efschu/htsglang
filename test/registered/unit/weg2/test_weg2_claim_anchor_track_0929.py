"""CLAIM ANCHOR (NF dynpf boot 0929_102750, 424346f693, rid weg2-24-38).

P prefilled a 16449-token prompt under ``--p-chunk-policy dynamic`` as
[0, 8192) + [8192, 16449). D read it back capped at 8192 of 16384 tokens::

    #1028B FETCH CAP n=7: kv=256 claimed=128 lost=128
        caps={mamba: 128, qsa_indexer: 256}
        anchors_in_range={mamba: (1, 127), qsa_indexer: (256, 255)}

The QSA index was complete (256/256); the cut is the MAMBA anchor. D's store
read ends at the deepest page a bigram reader can claim: the upstream match
leaves one token to forward (N-1 raw tokens) and a bigram key of r raw tokens
holds r-1 units, so 16448 raw -> 16447 units -> 256 pages = 16384. P's last
extend step tracked its recurrent state at floor_page(16449) = 16448, one page
past that claim, and the only anchor inside it was the chunk boundary at 8192.
With fixed 16k chunks the same prompt is [0, 16384) + [16384, 16449): the chunk
boundary happens to sit at 16384 and D reads 256/256 (weg2-10-14 in the same
boot, ``X-GATE uncached=65``) -- the fixed policy hid the defect for exactly
this probe length, it did not fix it (N = 20033 loses 3584 tokens there).

Hermetic, CPU. Pinned:
  * the reader's claim depth equals what the REAL upstream functions reach
    (``Req._compute_max_prefix_len`` + ``RadixKey.page_aligned``), bigram and
    unigram;
  * group P's extend step that crosses that claim tracks AT it (a mid-step
    grid point, +1 = the kernel reads h) -- the metal case and N % page == 0;
  * unchanged: a step starting at the claim (fixed 16k chunks), a default
    track at or below the claim, group D, no group, the upstream keying,
    page_size 1, a P-trim request.
"""

from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.weg2 import p_trim_end_anchor as pt
from sglang.srt.weg2 import tail_handoff as th

PAGE = 64


def _upstream_claim(n, page, bigram):
    """The deepest prefix a reader of an n-token prompt asks the store for."""
    max_prefix = Req._compute_max_prefix_len(
        SimpleNamespace(return_logprob=False, logprob_start_len=0), n)
    return len(RadixKey(array("q", range(n))[:max_prefix], is_bigram=bigram).page_aligned(page))


@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
@pytest.mark.parametrize("n", [65, 66, 128, 129, 130, 16448, 16449, 16450, 17247, 17248, 20033])
def test_the_reader_claim_is_what_the_upstream_prefetch_key_reaches(n, bigram):
    assert th.reader_claim_end(n, PAGE, bigram) == _upstream_claim(n, PAGE, bigram)


@pytest.fixture
def p_track(monkeypatch):
    import sglang.srt.managers.schedule_batch as sb

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.delenv("SGLANG_WEG2_FORK_ANCHOR_TOKEN", raising=False)
    monkeypatch.setattr(sb, "get_server_args", lambda: SimpleNamespace(
        mamba_cache_chunk_size=PAGE, mamba_checkpoint_interval=None,
        enable_mamba_extra_buffer_lazy=lambda: False))
    return monkeypatch, sb


def _track(sb, n, prefix, *, page=PAGE, bigram_exact=True, trim=False, rid="weg2-24-38"):
    tree = SimpleNamespace(page_size=page, bigram_anchor_exact=bigram_exact)
    batch = SimpleNamespace(
        tree_cache=tree,
        req_to_token_pool=SimpleNamespace(get_mamba_ping_pong_other_idx=lambda i: 1 - i))
    req = SimpleNamespace(rid=rid, origin_input_ids=array("q", range(n)), output_ids=[],
                          return_logprob=False, input_embeds=None, session_id=None,
                          multimodal_inputs=None, prefix_indices=list(range(prefix)),
                          mamba_ping_pong_track_buffer=torch.tensor([0, 1]), mamba_next_track_idx=0,
                          mamba_branching_seqlen=None, mamba_last_track_seqlen=None)
    if trim:
        setattr(req, pt.TRIM_ATTR, array("q", [n]))
    req.extend_range = SimpleNamespace(start=prefix, end=n, length=n - prefix)
    entry = sb.ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(batch, req)
    return req.mamba_last_track_seqlen, entry.track_seqlen


def test_the_dynpf_last_step_tracks_at_the_readers_claim(p_track):
    """weg2-24-38: [8192, 16449) tracked at 16448 -- D claims 16384."""
    _mp, sb = p_track
    aligned, seqlen = _track(sb, 16449, 8192)
    assert aligned == 16384 == th.reader_claim_end(16449, PAGE, True), aligned
    assert seqlen == 16385, "a mid-step grid point reads h (+1), like the fork track"


def test_every_dynpf_width_of_the_metal_prompt_lands_on_the_claim(p_track):
    """weg2-24-40/-41 of the same flip ended their last step elsewhere (16256,
    12160) -- the reads stopped at 254/190 of 256 pages; any start below the
    claim now tracks at it."""
    _mp, sb = p_track
    for start in (0, 8192, 12160, 16256, 16320):
        assert _track(sb, 16449, start)[0] == 16384, start


def test_a_page_multiple_prompt_also_tracks_at_the_claim(p_track):
    _mp, sb = p_track
    assert _track(sb, 16448, 8192) == (16384, 16385)


def test_the_fixed_chunk_step_is_unchanged(p_track):
    """weg2-10-14: [16384, 16449) -- the claim IS the step start, the chunk
    boundary before it carries the anchor; nothing moves."""
    _mp, sb = p_track
    assert _track(sb, 16449, 16384) == (16448, 16449)


@pytest.mark.parametrize("n,prefix", [(17248, 16640), (18784, 16640), (16510, 8192)])
def test_a_track_at_or_below_the_claim_is_unchanged(p_track, n, prefix):
    _mp, sb = p_track
    default = prefix + (n - prefix) // PAGE * PAGE
    assert default <= th.reader_claim_end(n, PAGE, True)
    assert _track(sb, n, prefix) == (default, n)


def test_only_group_p_moves_its_track(p_track):
    mp, sb = p_track
    mp.setenv("SGLANG_WEG2_GROUP", "D")
    assert _track(sb, 16449, 8192) == (16448, 16449), "group D: today's track"
    mp.delenv("SGLANG_WEG2_GROUP", raising=False)
    assert _track(sb, 16449, 8192) == (16448, 16449), "no group: today's track"


def test_the_keying_gates_the_rule(p_track):
    _mp, sb = p_track
    assert _track(sb, 16449, 8192, bigram_exact=False) == (16448, 16449), "upstream keying (27B)"
    assert _track(sb, 16449, 8192, page=1) == (16448, 16449), "page 1: no page to miss"
    assert _track(sb, 16449, 8192, trim=True) == (16448, 16449), "P-TRIM keeps its N-1 geometry"
