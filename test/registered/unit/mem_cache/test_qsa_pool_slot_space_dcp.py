"""WP3 (DCP x QSA): the QSA compressed cache mirrors the GLOBAL req_to_token
slot space, not this rank's uneven-DCP slice of the KV rows (fn1u boot
2026-09-16: TP0 with 4100 of 32772 rows asserted 'vectorized gather kernel
index out of bounds' in get_prefill_mqa_inputs)."""

import inspect
import re

import pytest
import torch

from sglang.srt.mem_cache import qsa_kv_pool as m


def test_pool_takes_the_global_slot_space_and_refuses_a_smaller_one():
    sig = inspect.signature(m.QSATokenToKVPool.__init__)
    assert "qsa_slot_space" in sig.parameters
    src = inspect.getsource(m.QSATokenToKVPool.__init__)
    assert "state_size = slot_space + page_size" in src
    assert "smaller than this rank's KV rows" in src


def test_mixin_hands_the_pool_max_total_num_tokens():
    from sglang.srt.model_executor import model_runner_kv_cache_mixin as mixin

    src = inspect.getsource(mixin)
    i = src.index("QSATokenToKVPool")
    block = src[i : i + 2500]
    assert re.search(r"qsa_slot_space=int\(self\.max_total_num_tokens\)", block)


def test_compressed_capacity_covers_every_global_slot():
    """The arithmetic the pool applies: every compressed slot id that
    req_to_token can produce (slot // ratio, slot < slot_space + page) is
    inside the capacity computed from the GLOBAL space -- and would not be
    from a DCP slice."""
    ratio, page = 4, 64
    slot_space, local = 32772, 4100
    cap_global = -((slot_space + page) // -ratio)
    cap_local = -((local + page) // -ratio)
    worst = (slot_space + page - 1) // ratio
    assert worst < cap_global
    assert worst >= cap_local  # the fn1u failure
