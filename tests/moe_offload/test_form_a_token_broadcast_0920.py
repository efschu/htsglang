"""fnFA10 (20.09. 13:32Z): the first Form A boot reached the server and died
in the workers' sampling -- a worker's forward yields no logits (MoE route
only). The host's sampled tokens reach the workers the way the weightless
lane does it: gloo broadcast on the generation step, the rank-0 accept
broadcast on a verify step."""

import inspect

from sglang.srt import rank_role
from sglang.srt.rank_role import HOST, WORKER, RankRolePlan, set_form_a_role_plan

FORM_A = RankRolePlan((HOST, WORKER, WORKER))


def test_host_predicate_and_token_source():
    try:
        set_form_a_role_plan(FORM_A, rank=0)
        assert rank_role.this_rank_is_form_a_host()
        assert not rank_role.this_rank_is_form_a_worker()
        assert rank_role.form_a_token_src_rank() == 0
        set_form_a_role_plan(FORM_A, rank=2)
        assert not rank_role.this_rank_is_form_a_host()
        assert rank_role.form_a_token_src_rank() == 0
    finally:
        set_form_a_role_plan(None)
    assert not rank_role.this_rank_is_form_a_host()
    assert rank_role.form_a_token_src_rank() is None


def test_generation_step_wires_worker_recv_and_host_send():
    from sglang.srt.managers import tp_worker as tw

    src = inspect.getsource(tw.TpModelWorker.forward_batch_generation)
    recv = src.index("head_ids = broadcast_pyobj(")
    assert 'getattr(self.model_runner, "is_form_a_worker", False)' in src[recv - 900 : recv]
    assert "form_a_token_src_rank()" in src[recv - 900 : recv]
    send = src.index("batch_result.next_token_ids.tolist(),")
    assert "this_rank_is_form_a_host()" in src[send - 900 : send]
    # the recv comes after the is_verify early return (ordering #143)
    assert src.index("if is_verify:") < recv


def test_verify_step_worker_receives_the_accept_broadcast():
    from sglang.srt.speculative import eagle_worker_v2 as ew

    src = inspect.getsource(ew)
    i = src.index("weightless_recv=_wl_worker")
    window = src[i - 1500 : i]
    assert '"is_form_a_worker", False' in window
