"""fnFL2x83 (23.09.): the front issues the waker's kv wake the moment the
waker's weights leg returns, not after the sleeper's release leg.

Bug regression. x82: D's weights leg returned at 20:26:10,170, P's release
leg at 10,457 (P's tail after its last pause: lane drains, empty_cache,
residue census, malloc_trim, DC breakdown, fence), and the kv wake was
issued only after BOTH -- 0,29 s of P's bookkeeping on D's first-token path
(WAKE-TAIL 10,169 -> CTRL-RECV of the kv RPC 10,459 on TP0). The kv pool's
VRAM is funded before D's leg returns (P's kv_cache is paused first, every
swapped P tag is paused by then). Hermetic: source bookkeeping of the flip
driver -- the async driver cannot run without two live groups.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front  # noqa: E402


def _src():
    return open(front.__file__).read()


def test_the_kv_wake_chains_on_the_wakers_leg_before_the_sleepers_leg_is_awaited():
    src = _src()
    i = src.index("_leg_tasks = [asyncio.ensure_future(c) for c in _legs]")
    blk = src[i:i + 900]
    a = blk.index("(w_code, w_body, w_ms) = await _w_task")
    b = blk.index("_kv_task = asyncio.ensure_future(self.leg_rpc(")
    c = blk.index("_res = await asyncio.gather(*_leg_tasks[:-1])")
    assert a < b < c, "the kv wake must be issued after the waker's leg and before the sleeper's is awaited"
    # the waker's leg is the LAST entry of `_legs` (sleep first, wake second)
    assert "_w_task = _leg_tasks[-1]" in blk


def test_step_five_takes_the_chained_answer_and_the_resident_form_still_issues_its_own():
    src = _src()
    j = src.index('self._flip_stage = "wake-kv"')
    blk = src[j:j + 700]
    assert "if _kv_task is not None:" in blk and "code, body = await _kv_task" in blk
    assert 'await self.leg_rpc(D, "/resume_memory_occupation"' in blk
    # `_kv_task` exists on the resident branch too (no NameError at step 5)
    k = src.index("_kv_task = None   # fnFL2x83")
    assert k < src.index("if self.weights_resident:", k)
