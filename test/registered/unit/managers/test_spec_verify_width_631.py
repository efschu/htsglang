"""#631: the verify's OUTPUT stride is the width that RAN.

THE DEFECT THIS PINS, measured on metal 2026-08-09 10:13:17Z. A request
carried across a PP->TP cutover runs one BOOTSTRAP round: a 1-node
trivial verify on an instance configured for 4 draft tokens. Its
``predict`` tensor therefore has ONE row per request, while
``_resolve_spec_v2_tokens`` sliced each request's accepted run at
``[i * stride, i * stride + accept_len)`` with ``stride`` taken from the
instance's CONFIGURED width. Request 0 sliced its single token
correctly; every later request sliced past the end of the tensor and got
an EMPTY LIST, and ``req.output_ids.extend([])`` appends nothing.

All three ranks logged it identically, which is what ruled out every
transport suspect:

    round kind=decode -- 3367da51 have=49 +[220]
                       | bcf3eb14 have=49 +[]
                       | 6772dc40 have=49 +[]

The KV had advanced, so the answer resumed one token short rather than
diverging: "...19 2021" where "19 20 21" was due.

THE FIX IS AT THE SOURCE, not at this consumer: ``verify`` stamps the
result with ``verify_input.draft_token_num``. Three consumers divide by
that field (this one, the adaptive controller's rung attribution, and the
``return_hidden_states`` lane), and a per-consumer patch would have left
the other two wrong. It is the same general form as the
``_draft_extend_for_decode`` width defect fixed in 4147972205 -- one
consumer further along.
"""

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)


class _Req:
    def __init__(self):
        self.is_retracted = False
        self.grammar = None
        self.kv_committed_len = 0
        self.spec_verify_ct = 0
        self.spec_num_correct_drafts = 0
        self.hist = []

    def finished(self):
        return False

    def update_spec_correct_drafts_histogram(self, n):
        self.hist.append(n)


class _Batch:
    def __init__(self, n):
        self.reqs = [_Req() for _ in range(n)]


class _Result:
    def __init__(self, tokens, accept_lens, stride):
        self.next_token_ids = torch.tensor(tokens, dtype=torch.int64)
        self.accept_lens = torch.tensor(accept_lens, dtype=torch.int64)
        self.block_accept_lens = None
        self.cap_lens = None
        self.speculative_num_draft_tokens = stride


class _Worker:
    def __init__(self):
        self.calls = []

    def on_verify_complete_cpu(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _Self:
    """Duck-typed processor: _resolve_spec_v2_tokens touches only these."""

    def __init__(self):
        self.model_worker = _Worker()

    def _accept_grammar_tokens(self, req, tokens):
        return tokens


def _resolve(result, batch):
    return SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
        _Self(), result, batch
    )


# --------------------------------------------------------------------
# The bootstrap round: bs rows, one per request, width 1
# --------------------------------------------------------------------


def test_narrowed_round_gives_every_request_its_token():
    """THE FIX. One row per request and a stride of 1: all three land."""
    batch = _Batch(3)
    result = _Result([220, 220, 220], [1, 1, 1], stride=1)
    assert _resolve(result, batch) == [[220], [220], [220]]


def test_can_fail_the_configured_stride_empties_every_request_but_the_first():
    """THE DEFECT, reproduced exactly. Same three rows, stride 4 -- the
    instance's configured width -- and requests 1 and 2 slice past the end
    of the tensor. This is the metal specimen in unit form, and it is the
    proof that the pin above can fail."""
    batch = _Batch(3)
    result = _Result([220, 220, 220], [1, 1, 1], stride=4)
    assert _resolve(result, batch) == [[220], [], []]


def test_an_ordinary_full_width_round_is_unchanged():
    """The two widths are equal on every ordinary round, which is why the
    static value stood in for years. Pinned so the fix cannot regress it."""
    batch = _Batch(2)
    tokens = [10, 11, 12, 13, 20, 21, 22, 23]
    result = _Result(tokens, [3, 2], stride=4)
    assert _resolve(result, batch) == [[10, 11, 12], [20, 21]]


def test_accounting_follows_the_accepted_run():
    batch = _Batch(2)
    result = _Result([10, 11, 12, 13, 20, 21, 22, 23], [3, 2], stride=4)
    _resolve(result, batch)
    assert [r.kv_committed_len for r in batch.reqs] == [3, 2]
    assert [r.spec_verify_ct for r in batch.reqs] == [1, 1]
    assert [r.spec_num_correct_drafts for r in batch.reqs] == [2, 1]
    assert result.num_correct_drafts == 3


def test_narrowed_round_reports_zero_correct_drafts():
    """A 1-node verify accepts its root and drafts nothing, so the
    bootstrap round must not be credited with correct drafts."""
    batch = _Batch(3)
    result = _Result([220, 220, 220], [1, 1, 1], stride=1)
    _resolve(result, batch)
    assert result.num_correct_drafts == 0
    assert result.num_correct_drafts_per_req_cpu == [0, 0, 0]
