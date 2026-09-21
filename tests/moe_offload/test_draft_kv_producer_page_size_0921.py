"""#66 (fnFL2v65, 21.09.): the draft-KV producer accepts page_size > 1.

The producer's parse-time scope refused ``page_size != 1`` with "a canonical
draft page is ONE token's draft layer".  Nothing in the path holds that:
``page_size`` appears nowhere in draft_kv_producer.py, the draft write lands
on the TARGET batch's own ``out_cache_loc`` slots, the canonical store sizes
its draft window per token, and the only ``2 * kv_heads * head_dim`` in the
tree is the producer's LOG LINE.  Next Flash runs page_size 64, so the
refusal left the decode group's drafter unmatched (W10, P=None) and the flip
had no draft KV to carry.

The OTHER refusals in that scope are real and must stay.
"""

import pytest

from sglang.srt.server_args import ServerArgs


def _args(**over):
    a = ServerArgs.__new__(ServerArgs)
    base = dict(
        speculative_draft_kv_only=True,
        pp_size=3,
        tp_size=1,
        dp_size=1,
        ep_size=1,
        page_size=64,
        hicache_canonical_kv_page=True,
        speculative_algorithm="NEXTN",
        speculative_num_steps=2,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=3,
        speculative_draft_placement="split",
        speculative_cross_algorithm=None,
    )
    base.update(over)
    for k, v in base.items():
        setattr(a, k, v)
    return a


@pytest.mark.parametrize("page_size", [1, 16, 64, 128])
def test_every_page_size_is_accepted(page_size):
    _args(page_size=page_size)._handle_speculative_draft_kv_only()


def test_the_flag_off_is_a_no_op():
    _args(speculative_draft_kv_only=False, pp_size=1)._handle_speculative_draft_kv_only()


@pytest.mark.parametrize(
    "over,needle",
    [
        (dict(pp_size=1), "pp_size > 1"),
        (dict(tp_size=2), "tp_size == 1"),
        (dict(dp_size=2), "data parallelism"),
        (dict(ep_size=2), "expert parallelism"),
        (dict(hicache_canonical_kv_page=False), "canonical-kv-page"),
        (dict(speculative_num_steps=None), "speculative-num-steps"),
        (dict(speculative_draft_placement="solo"), "solo"),
        (dict(speculative_cross_algorithm="x"), "cross-algorithm"),
    ],
)
def test_the_real_refusals_stay(over, needle):
    with pytest.raises(ValueError, match=needle):
        _args(**over)._handle_speculative_draft_kv_only()


def test_the_producer_log_line_counts_a_whole_page():
    """An instrument that reports one token's bytes under page_size 64
    misreads the store by that factor."""
    import inspect

    from sglang.srt.managers import scheduler as sch

    src = inspect.getsource(sch.Scheduler._maybe_init_draft_kv_producer)
    assert 'getattr(self.server_args, "page_size", 1)' in src
