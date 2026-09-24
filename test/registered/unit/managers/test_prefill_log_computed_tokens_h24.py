"""fnFL2 H24: the prefill lines count the tokens a forward COMPUTES.

Bug (metal fnFL2x137, D TP0): the adopted 1-token extend [97840, 97841) was
logged as 'Prefill rank batch, #new-token: 64, #cached-token: 97840' and
'Prefill batch ... #new-token: 64 ... input throughput (token/s): 1.10'
while the forward computed ONE row (#998 EXTENT (97840, 97841, 1),
HOST-ANON-PASS phase=EXTEND tokens=1). The whole-fit admission hands
``_update_prefill_budget`` the page-CEILED length (it is the budget's unit),
and the same number fed ``log_input_tokens`` -- so every tail shorter than a
page read as a full page, and the rank line looked like a padded forward.
The budget keeps the ceiled charge; only the log counters change.
"""

import inspect

from sglang.srt.managers.schedule_policy import PrefillAdder


def _adder(page=64):
    a = PrefillAdder.__new__(PrefillAdder)
    a.page_size = page
    a.rem_total_token_offset = a.cur_rem_token_offset = 0
    a.rem_mamba_slots = None
    a.rem_input_tokens = 1 << 20
    a.is_hybrid_swa = False
    a.dllm_config = None
    a.rem_chunk_tokens = None
    a.log_hit_tokens = a.log_input_tokens = 0
    a.reprocessed_log_hit_tokens = a.reprocessed_log_input_tokens = 0
    return a


def test_log_counts_the_computed_extend_budget_the_page():
    a = _adder()
    a._update_prefill_budget(97840, 64, 0, True, computed_input_len=1)
    assert (a.log_input_tokens, a.reprocessed_log_input_tokens, a.log_hit_tokens) == (1, 1, 97840)
    assert a.cur_rem_token_offset == 64 + 64  # budget: ceiled extend + page overhead, unchanged
    assert a.rem_input_tokens == (1 << 20) - 64


def test_without_the_computed_length_the_caller_value_is_logged():
    a = _adder()
    a._update_prefill_budget(0, 49, 0, False)  # a caller that passes the raw length
    assert a.log_input_tokens == 49 and a.cur_rem_token_offset == 64 + 64


def test_the_whole_fit_admission_names_the_computed_length():
    """The one call site that passes the ceiled ``input_tokens`` must name
    the computed extend -- dropping the keyword brings '#new-token: 64' back."""
    src = inspect.getsource(PrefillAdder.add_one_req)
    assert "computed_input_len=_ea_len if _ea_forced else len(req.full_untruncated_fill_ids) - _ea_start" in src
