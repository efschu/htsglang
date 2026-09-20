"""fnFA2 (20.09.): the 'TP > num_kv_heads requires uneven DCP' refusal in the
attention layer must be waived under Form A -- the host holds every kv head
once (no replication across ranks), DCP is off by design, and a worker never
attends."""

import inspect


def test_the_refusal_is_waived_when_the_host_owns_every_head():
    from sglang.srt.models import qwen3_5 as m

    src = inspect.getsource(m)
    i = src.index("TP > num_kv_heads: layer")
    window = src[i - 700 : i]
    assert "not form_a_dense_is_unsharded()" in window
    assert "get_parallel().attn_dcp_size != self.attn_tp_size" in window


def test_form_a_dense_is_unsharded_is_false_on_a_classic_boot():
    from sglang.srt.rank_role import form_a_dense_is_unsharded

    assert form_a_dense_is_unsharded() is False
