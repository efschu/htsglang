import pathlib
"""Fail-fast guard: the MLA DCP collectives have no uneven-TP variant.

``cp_lse_ag_out_rs_mla`` and ``all_gather_q_for_mla_decode`` both assume an
EQUAL per-rank head count. Under a non-uniform --rank-tp-ratio they do not
error by themselves -- they either compute a silently wrong result or deadlock,
because torch's reduce_scatter/all_gather require every rank to agree on the
shape. This suite pins the rejection, and pins that the guard stays a no-op on
the stock (even / unset) paths so nothing existing changes behaviour.

CPU only: the guard runs before any collective or device work.
"""

import unittest

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    set_tp_partition_ratios,
)
from sglang.srt.layers.dcp.comm import _reject_uneven_tp_mla
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MLA_FNS = (
    ("cp_lse_ag_out_rs_mla", "Its reduce_scatter_along_dim(dim=0) ..."),
    ("all_gather_q_for_mla_decode", "It all-gathers a [H, B, L] tensor ..."),
)


class TestUnevenTpMlaGuard(CustomTestCase):
    def setUp(self):
        self._saved = get_tp_partition_ratios()

    def tearDown(self):
        set_tp_partition_ratios(self._saved)

    def test_noop_when_ratios_unset(self):
        """Stock path: no ratio vector installed -> guard must not fire."""
        set_tp_partition_ratios(None)
        for fn, detail in MLA_FNS:
            _reject_uneven_tp_mla(fn, detail)

    def test_noop_on_even_ratios(self):
        """An all-equal vector is the even split -> guard must not fire."""
        for even in ([1, 1], [1, 1, 1], [4, 4, 4, 4]):
            set_tp_partition_ratios(even)
            for fn, detail in MLA_FNS:
                _reject_uneven_tp_mla(fn, detail)

    def test_rejects_uneven_ratios(self):
        """Non-uniform vector -> hard rejection for both MLA collectives."""
        for uneven in ([2, 1], [2, 1, 1], [12, 6, 6]):
            set_tp_partition_ratios(uneven)
            for fn, detail in MLA_FNS:
                with self.assertRaises(NotImplementedError) as cm:
                    _reject_uneven_tp_mla(fn, detail)
                msg = str(cm.exception)
                # The message must name WHAT is unsupported and WHY.
                self.assertIn(fn, msg)
                self.assertIn("uneven tensor parallelism", msg)
                self.assertIn(str(uneven), msg)
                self.assertIn("no uneven-MLA variant", msg)

    def test_decision_is_a_pure_function_of_the_ratio_vector(self):
        """RANK-UNIFORMITY, the actual point of the guard.

        The predicate reads only process-global state that every rank derives
        deterministically from the same CLI args, so all ranks reach the same
        verdict. A guard that fired on a subset of ranks would itself be a hang
        source -- exactly what it exists to prevent. Assert determinism and
        absence of hidden state by re-evaluating repeatedly and by flipping
        back and forth between vectors.
        """
        for _ in range(3):
            set_tp_partition_ratios([2, 1])
            for fn, detail in MLA_FNS:
                with self.assertRaises(NotImplementedError):
                    _reject_uneven_tp_mla(fn, detail)
            set_tp_partition_ratios([1, 1])
            for fn, detail in MLA_FNS:
                _reject_uneven_tp_mla(fn, detail)


if __name__ == "__main__":
    unittest.main()


def test_even_dcp_triton_requires_replicated_kv_heads_across_the_group():
    """The even-DCP triton decode gathers the DCP group's q heads and attends
    them against the LOCAL kv-head shard; the kernel remaps q->kv as
    cur_head // (gathered_q // local_kv). Correct ONLY when every rank of the
    group holds the same full kv-head set (tp // total_kv >= dcp_size).

    Measured outside that geometry (Qwen2.5-1.5B, q12/kv2, TP=2/DCP=2): CJK
    mojibake in an English greedy completion from the FIRST decode token, on
    NCCL and HTCCL alike; the same model at TP=4/DCP=2 (replicas 2 >= 2) is
    coherent, and a 1-token prompt still garbles -- so it is the head
    geometry, not the token-ownership layout. This pins the boot-time guard
    that converts that silent garbage into a startup error.
    """
    import re

    import sglang.srt.layers.attention.triton_backend as tb

    src = pathlib.Path(tb.__file__).read_text()
    # the guard must sit in the constructor, gated on even DCP only
    m = re.search(
        r"if self\.dcp_size > 1 and not uneven_dcp_active\(self\.dcp_size\):"
        r"[\s\S]{0,900}?raise ValueError",
        src,
    )
    assert m, (
        "the even-DCP kv-head replication guard is gone from "
        "TritonAttnBackend.__init__; Qwen2/Qwen3-class models under "
        "--attention-backend triton --dcp-size N would emit silent garbage "
        "again"
    )
    body = m.group(0)
    assert "get_total_num_kv_heads" in body
    assert "dcp_size" in body

    # arithmetic of the condition itself, on the measured cases
    cases = [
        # (attn_tp, total_kv, dcp, must_reject)
        (2, 2, 2, True),    # Qwen2.5-1.5B TP=2/DCP=2 -> garbage, must reject
        (2, 8, 2, True),    # Qwen3-0.6B                 -> garbage, must reject
        (4, 2, 2, False),   # replicas 2 >= 2            -> measured coherent
        (8, 4, 2, False),   # the path's origin geometry (Qwen3.5-397B)
    ]
    for attn_tp, kv, dcp, must_reject in cases:
        replicas = attn_tp // kv if kv else 0
        assert (replicas < dcp) == must_reject, (attn_tp, kv, dcp)
