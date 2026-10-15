# SPDX-License-Identifier: Apache-2.0
"""K11 (NF h16p 226aad7105, 15.10.2026 12:39:42Z): a budget-bound continuation poisoned the next QSA prefix.

The death (boot ..._226aad7105_1015_122948, P.log 29069/29397): PP0's pool was 94 % full, ``add_chunked_req`` took
``min(rem_chunk_tokens, rem_total_tokens)`` = 9586 raw, the chunk [48960, 58546) ended off the QSA grain, and the next
chunk's prefix 58546 (% 4 == 2) fired the text-less ``torch._assert_async((prefix_lens % ratio == 0).all())`` in
``_qsa_build_write_plan`` -- reported asynchronously at hybrid_linear_attn_backend.py:301.

GREEN: (1) a truncated chunk ends on the end-anchor grain (page, or the ratio under the tail hand-off; 1 off QSA);
a budget below the next boundary is the #679 park. (2) the QSA write plan stops with a named RuntimeError on the host
lengths before any kernel, and both device asserts carry text.

Hermetic: mocks only, no CUDA.
"""

import inspect
import os
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, PropertyMock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa  # noqa: E402
from sglang.srt.managers import schedule_policy as sp  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult  # noqa: E402
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool  # noqa: E402
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402
from sglang.srt.utils.common import Range  # noqa: E402

PAGE = 64
RATIO = 4
CHUNK = 16384
PREFIX = 48960  # weg2-3-20, P.log 29069
BUDGET = 9586  # the raw rem_total_tokens that ended the chunk at 58546
FILL = 121219  # 58546 + 62673 pending


def _qsa_pool():
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    return kv


def _adder(*, qsa_pool=True, rem_chunk=CHUNK):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    tree = MagicMock()
    tree.full_evictable_size.return_value = 0
    tree.swa_evictable_size.return_value = 0
    tree.evictable_size.return_value = 0
    tree.disable = False
    tree.inc_lock_ref.return_value = IncLockRefResult()
    tree.dec_lock_ref.return_value = DecLockRefResult()
    alloc = MagicMock()
    alloc.available_size.return_value = 10**6
    alloc.full_available_size.return_value = 10**6
    if qsa_pool:
        alloc.get_kvcache.return_value = _qsa_pool()
    running = MagicMock()
    running.reqs = []
    return PrefillAdder(
        page_size=PAGE,
        tree_cache=tree,
        token_to_kv_pool_allocator=alloc,
        running_batch=running,
        new_token_ratio=1.0,
        rem_input_tokens=10**6,
        rem_chunk_tokens=rem_chunk,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )


def _req(prefix=PREFIX, fill=FILL, rid="weg2-3-20"):
    req = MagicMock(spec=Req)
    req.rid = rid
    req.priority = 0
    req.prefix_indices = list(range(prefix))
    req.full_untruncated_fill_ids = list(range(fill))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.set_extend_range = MagicMock(
        side_effect=lambda start, end: setattr(req, "extend_range", Range(start, end))
    )
    return req


def _run_chunked(adder, req, budget):
    with mock.patch.object(PrefillAdder, "rem_total_tokens", new_callable=PropertyMock, return_value=budget):
        return adder.add_chunked_req(req)


@pytest.fixture(params=["page", "ratio"])
def grain(request, monkeypatch):
    # tail hand-off off -> the page; on -> the compress ratio (tail_handoff.anchor_grain)
    monkeypatch.setenv("SGLANG_WEG2_TAIL_HANDOFF", "1" if request.param == "ratio" else "0")
    return PAGE if request.param == "page" else RATIO


def test_grain_is_qsa_only(grain):
    assert _adder(qsa_pool=True)._weg2_chunk_grain() == grain
    assert _adder(qsa_pool=False)._weg2_chunk_grain() == 1  # a non-QSA pool: unchanged behaviour
    no_kv = _adder(qsa_pool=False)
    no_kv.token_to_kv_pool_allocator = SimpleNamespace()  # desk double without get_kvcache
    assert no_kv._weg2_chunk_grain() == 1


def test_k11_budget_bound_chunk_ends_on_grain(grain):
    """The incident numbers: prefix 48960, budget 9586 -> the end lands on the grain, never on 58546."""
    adder, req = _adder(), _req()
    out = _run_chunked(adder, req, BUDGET)
    assert out is req  # still chunked
    start, end = req.extend_range.start, req.extend_range.end
    assert start == PREFIX
    assert end != PREFIX + BUDGET
    assert end % grain == 0 and end % RATIO == 0
    assert end == (PREFIX + BUDGET) // grain * grain  # 58496 (page) / 58544 (ratio)
    assert req in adder.can_run_list
    # the NEXT chunk's prefix is that end: aligned, so the QSA host check passes
    nxt = SimpleNamespace(seq_lens_cpu=torch.tensor([end + CHUNK]), extend_seq_lens_cpu=[CHUNK], rids=[req.rid])
    qsa.qsa_check_prefix_grain(nxt, RATIO, 1)


def test_k11_budget_below_grain_parks(grain):
    """A budget that does not reach the next boundary is the #679 park, not a sub-grain chunk."""
    adder, req = _adder(), _req()
    out = _run_chunked(adder, req, grain - 1)
    assert out is req
    assert (req.extend_range.start, req.extend_range.end) == (PREFIX, PREFIX)
    assert req not in adder.can_run_list


def test_non_qsa_unchanged():
    """Off QSA the raw budget still decides (grain 1): byte-for-byte the old chunk end."""
    adder, req = _adder(qsa_pool=False), _req()
    _run_chunked(adder, req, BUDGET)
    assert (req.extend_range.start, req.extend_range.end) == (PREFIX, PREFIX + BUDGET)


def test_last_chunk_not_floored(grain):
    """The final chunk (not truncated) keeps its real end; only a chunk that stays chunked is floored."""
    adder, req = _adder(), _req(fill=PREFIX + 3150)
    assert not sp._WEG2_END_ANCHOR  # the end-anchor split is the launcher's (off here); not this test's subject
    out = _run_chunked(adder, req, BUDGET)
    assert out is None
    assert (req.extend_range.start, req.extend_range.end) == (PREFIX, PREFIX + 3150)


def test_ignore_eos_chunk_floored(grain):
    """The second mint site with a raw budget: a leftover rem_chunk_tokens (9586) is floored the same way."""
    adder = _adder(rem_chunk=BUDGET)
    req = _req(prefix=0, fill=20000)
    req.origin_input_ids = list(range(20000))
    req.sampling_params.ignore_eos = True
    req.host_hit_length = 0
    with mock.patch.object(PrefillAdder, "rem_total_tokens", new_callable=PropertyMock, return_value=10**6):
        adder.add_one_req_ignore_eos(req)
    assert req in adder.can_run_list
    assert (req.extend_range.start, req.extend_range.end) == (0, BUDGET // grain * grain)


def test_floor_after_the_told_extent():
    """Followers under the row authority run `_add_scheduled_req` (PP0's extent) and return before the floor:
    they never cut on their own pool, so every stage gets PP0's end."""
    src = inspect.getsource(PrefillAdder.add_chunked_req)
    assert src.index("scheduled_extent_for(") < src.index("_weg2_floor_chunk_to_grain(")
    assert src.index("_weg2_floor_chunk_to_grain(") < src.index("_weg2_p_fork_cut.apply(")


def test_named_stop_before_kernel():
    """The incident batch: prefix 58546 -> RuntimeError naming rid / prefix / ratio, on host lengths only."""
    fb = SimpleNamespace(seq_lens_cpu=torch.tensor([74930]), extend_seq_lens_cpu=[16384], rids=["weg2-3-20"])
    assert qsa.qsa_prefix_grain_violations(fb, RATIO, 1) == [(0, 58546)]
    with pytest.raises(RuntimeError, match=r"QSA-PREFIX-GRAIN rid=weg2-3-20 prefix=58546 ratio=4"):
        qsa.qsa_check_prefix_grain(fb, RATIO, 1)
    # scheduler's prefix list when the pair is missing
    fb2 = SimpleNamespace(seq_lens_cpu=None, extend_seq_lens_cpu=None, extend_prefix_lens_cpu=[58546], rids=None)
    with pytest.raises(RuntimeError, match="rid=\\?"):
        qsa.qsa_check_prefix_grain(fb2, RATIO, 1)
    # no host lengths -> silent (the device assert stays the backstop); aligned -> silent
    qsa.qsa_check_prefix_grain(SimpleNamespace(), RATIO, 1)
    qsa.qsa_check_prefix_grain(
        SimpleNamespace(seq_lens_cpu=torch.tensor([74880]), extend_seq_lens_cpu=[16384]), RATIO, 1
    )


def test_device_asserts_carry_text():
    src = inspect.getsource(qsa.QwenSparseAttnBackend._qsa_write_plan) + inspect.getsource(
        qsa.QwenSparseAttnBackend._qsa_build_write_plan
    )
    assert '"QSA-TABLE-WIDTH' in src and '"QSA-PREFIX-GRAIN' in src
    plan = inspect.getsource(qsa.QwenSparseAttnBackend._qsa_build_write_plan)
    assert plan.index("qsa_check_prefix_grain(") < plan.index("torch._assert_async(")
