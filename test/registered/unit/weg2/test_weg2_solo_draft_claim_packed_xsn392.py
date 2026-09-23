"""xsn392 (19.09.): the prefetch claim reduce must have ONE form on every
rank. Under --speculative-draft-placement solo the host arms the draft tier
(packed [hit, -hit] MIN) and the shadows do not (scalar MIN): a 2-vector
against two scalars timed every probe out at its budget with 0 pages on
every rank. A solo shadow answers the packed form.

fnFL2x22 (23.09.): the SAME signature on the Form A expert workers (null
storage tier, a shadow kind the xsn392 marker never covered): all three D
prefetch threads parked in the claim collective for the whole reap budget
(255 s), 0 pages on every rank, the 259392-slot host span of the first
prefetch leaked (available=66176 = 325568-259392), host_pool_exhausted, W17.
The form is now AGREED over the group (one scalar MAX) and a byteless tier
ABSTAINS from the packed vote instead of claiming every page."""
from __future__ import annotations

import os
import queue
import threading
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.cache_controller import (  # noqa: E402
    CLAIM_VOTE_ABSTAIN,
    HiCacheController,
    Weg2DraftDisagree,
    assert_draft_claims_agree,
    claim_vote_abstains,
    decode_claim_vote,
    encode_claim_vote,
)
from sglang.srt.mem_cache import kv_cache_builder as kb  # noqa: E402
from sglang.srt.mem_cache.hicache_collective import (  # noqa: E402
    HiCacheCollectiveTimeoutError,
)
from sglang.srt.mem_cache.hicache_storage import FormAWorkerNullStorage  # noqa: E402


def _stub(armed: bool, shadow):
    st = types.SimpleNamespace(draft_tier_armed=lambda direction: armed)
    if shadow is not None:
        st.solo_draft_shadow = shadow
    return st


def test_the_packed_form_follows_the_draft_tier_or_the_solo_shadow_mark():
    f = HiCacheController.draft_claim_packed
    assert f(_stub(True, None)) is True          # the host / any draft-armed rank
    assert f(_stub(False, None)) is False        # a plain rank without a draft
    assert f(_stub(False, False)) is False
    assert f(_stub(False, True)) is True         # the solo shadow


def test_the_builder_marks_a_solo_shadow_and_leaves_others_alone():
    class _Ctl:
        solo_draft_shadow = False
    ctl = _Ctl()
    tree = types.SimpleNamespace(cache_controller=ctl)
    runner = types.SimpleNamespace(token_to_kv_pool=None, is_draft_solo_shadow=True)
    dw = types.SimpleNamespace(draft_worker=types.SimpleNamespace(draft_runner=runner))
    sa = types.SimpleNamespace(enable_multi_layer_eagle=False)
    spec = types.SimpleNamespace(is_ngram=lambda: False)
    kb.maybe_register_hicache_draft(tree_cache=tree, draft_worker=dw, spec_algorithm=spec,
                                    server_args=sa, enable_hierarchical_cache=True, page_size=1)
    assert ctl.solo_draft_shadow is True
    ctl2 = _Ctl()
    runner2 = types.SimpleNamespace(token_to_kv_pool=None, is_draft_solo_shadow=False)
    dw2 = types.SimpleNamespace(draft_worker=types.SimpleNamespace(draft_runner=runner2))
    kb.maybe_register_hicache_draft(tree_cache=types.SimpleNamespace(cache_controller=ctl2),
                                    draft_worker=dw2, spec_algorithm=spec, server_args=sa,
                                    enable_hierarchical_cache=True, page_size=1)
    assert ctl2.solo_draft_shadow is False


# --- fnFL2x22: the form is a GROUP agreement, a byteless tier abstains -------


def _min_reduce(votes):
    """What the gloo MIN over the group returns: elementwise min of the votes."""
    return [min(v[i] for v in votes) for i in range(2)]


def test_x22_a_byteless_tier_abstains_and_the_voters_decide():
    """RED ON dfdb417326 (the null tier 'claims every page'): the host's
    anchor-capped claim 4052 beside the worker's 4053 was a rank disagreement
    -- under the packed form a STOP, never the MIN the host decides."""
    host = encode_claim_vote(4052, abstain=False)
    w1 = encode_claim_vote(4053, abstain=True)
    w2 = encode_claim_vote(4053, abstain=True)
    assert w1 == [CLAIM_VOTE_ABSTAIN, CLAIM_VOTE_ABSTAIN]
    mn, mx = decode_claim_vote(_min_reduce([host, w1, w2]))
    assert (mn, mx) == (4052, 4052)
    assert_draft_claims_agree(mn, mx, "rid")  # the host decides, no STOP
    # MUTANT (the pre-fix vote): a worker that VOTES its every-page claim.
    mn, mx = decode_claim_vote(_min_reduce([host, encode_claim_vote(4053, False)]))
    with pytest.raises(Weg2DraftDisagree):
        assert_draft_claims_agree(mn, mx, "rid")


def test_x22_a_group_of_abstainers_holds_nothing():
    votes = [encode_claim_vote(9, True), encode_claim_vote(7, True)]
    assert decode_claim_vote(_min_reduce(votes)) == (0, 0)


def test_x22_the_null_tier_says_it_abstains_and_a_real_tier_does_not():
    assert FormAWorkerNullStorage().abstains_from_claim_vote is True
    assert claim_vote_abstains(types.SimpleNamespace(storage_backend=FormAWorkerNullStorage()))
    assert not claim_vote_abstains(types.SimpleNamespace(storage_backend=object()))
    assert not claim_vote_abstains(types.SimpleNamespace(storage_backend=None))


def _form_stub(*, armed: bool, peer_packed: bool, groups: int):
    """A controller whose peers answer the form MAX with ``peer_packed``."""
    seen = []

    def _reduce(self, tensor, op):
        assert op == torch.distributed.ReduceOp.MAX
        seen.append(int(tensor[0]))
        tensor[0] = max(int(tensor[0]), int(peer_packed))

    st = types.SimpleNamespace(
        draft_tier_armed=lambda direction: armed,
        prefetch_sync_groups=[object()] * groups,
        storage_backend=None,
        _seen=seen,
    )
    st._all_reduce_prefetch_groups = types.MethodType(_reduce, st)
    return st


def test_x22_the_form_is_agreed_over_the_group_not_read_off_the_rank():
    """A worker without a draft tier adopts the packed form because a PEER
    needs it -- by agreement, not by a marker someone has to remember."""
    op = types.SimpleNamespace(request_id="rid")
    st = _form_stub(armed=False, peer_packed=True, groups=1)
    assert HiCacheController._agree_claim_form(st, op) is True
    assert st._seen == [0]  # this rank's own vote was 'scalar'
    st = _form_stub(armed=True, peer_packed=False, groups=1)
    assert HiCacheController._agree_claim_form(st, op) is True
    st = _form_stub(armed=False, peer_packed=False, groups=1)
    assert HiCacheController._agree_claim_form(st, op) is False


def test_x22_without_a_sync_group_the_local_answer_stands():
    """No group, no collective: the answer is this rank's own (single-rank
    processes, the tests' stubs) and no reduce is attempted."""
    op = types.SimpleNamespace(request_id="rid")
    st = _form_stub(armed=True, peer_packed=False, groups=0)
    assert HiCacheController._agree_claim_form(st, op) is True
    assert st._seen == []


def test_x22_a_claim_collective_that_expires_is_the_group_stop():
    """The unbounded reduce parked three threads for 255 s and the reaper
    read it as 'the store held nothing'. An expiry is now the S5 STOP."""
    stops = []
    stop = threading.Event()
    q = queue.Queue()
    q.put(types.SimpleNamespace(request_id="rid", mark_terminate=lambda: None))

    def _expire(self, operation):
        raise HiCacheCollectiveTimeoutError("prefetch-thread/all_reduce expired")

    st = types.SimpleNamespace(
        storage_stop_event=stop,
        prefetch_queue=q,
        prefetch_io_aux_func=lambda: None,
        _prefetch_drained_after_stop=0,
        _storage_hit_query=lambda op: (["k"], 64),
        _stop_group_from_thread=lambda exc: stops.append(type(exc).__name__),
    )
    st._agree_claim_form = types.MethodType(_expire, st)
    HiCacheController.prefetch_thread_func(st)
    assert stops == ["HiCacheCollectiveTimeoutError"]
