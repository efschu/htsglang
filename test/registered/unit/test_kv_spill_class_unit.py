"""Unit tests for the per-session kv-session-offload spill (latency) class.

Pure ordering helpers + the arg-parse gate. CPU-only; no server, no GPU. Run:
  python -m pytest test/registered/unit/test_kv_spill_class_unit.py -q
"""

import argparse

import pytest

from sglang.srt.managers.kv_session_offload import (
    SPILL_CLASS_NEVER,
    SPILL_CLASS_NORMAL,
    SPILL_CLASS_PREFERRED,
    SPILL_CLASSES,
    select_spill_victim,
    session_priority_key,
    spill_class_of,
)


class _FakeReq:
    def __init__(self, seq, fast=False, spill_class=None):
        self.kv_arrival_seq = seq
        self.is_fast_lane = fast
        if spill_class is not None:
            self.spill_class = spill_class

    def finished(self):
        return False


# -- class resolution -----------------------------------------------------


def test_spill_class_defaults_to_normal():
    # unset attribute, explicit None and an unknown string all resolve to
    # NORMAL -- the spill hot path must never raise on internal reqs that
    # never carried the field.
    assert spill_class_of(_FakeReq(1)) == SPILL_CLASS_NORMAL
    assert spill_class_of(_FakeReq(1, spill_class=None)) == SPILL_CLASS_NORMAL
    assert spill_class_of(_FakeReq(1, spill_class="bogus")) == SPILL_CLASS_NORMAL
    assert SPILL_CLASSES == (
        SPILL_CLASS_PREFERRED,
        SPILL_CLASS_NORMAL,
        SPILL_CLASS_NEVER,
    )


# -- default path is byte-identical --------------------------------------


def test_default_path_ordering_unchanged():
    """No class set anywhere -> the victim order is exactly the pre-class one.

    The pre-class key was (is_fast_lane, -arrival_seq); the class only
    prepends a CONSTANT rank when nothing is tagged, so every comparison is
    unchanged. Asserted through both the key and the selector."""
    plain = [_FakeReq(3), _FakeReq(9), _FakeReq(5), _FakeReq(11, fast=True)]
    tagged = [
        _FakeReq(3, spill_class=SPILL_CLASS_NORMAL),
        _FakeReq(9, spill_class=SPILL_CLASS_NORMAL),
        _FakeReq(5, spill_class=SPILL_CLASS_NORMAL),
        _FakeReq(11, fast=True, spill_class=SPILL_CLASS_NORMAL),
    ]
    for a, b in zip(plain, tagged):
        assert session_priority_key(a) == session_priority_key(b)
    for fp in (False, True):
        assert select_spill_victim(plain, fast_pressure=fp) == select_spill_victim(
            tagged, fast_pressure=fp
        )
    # unchanged against the documented pre-class expectation
    assert select_spill_victim(plain) == 1  # youngest normal
    assert select_spill_victim([_FakeReq(1)]) is None  # sole session tabu


# -- "never" is tabu ------------------------------------------------------


def test_never_is_never_a_victim():
    reqs = [
        _FakeReq(0),
        _FakeReq(9, spill_class=SPILL_CLASS_NEVER),
        _FakeReq(5),
    ]
    # youngest is the 'never' session -> the next-youngest normal loses instead
    assert select_spill_victim(reqs) == 2
    assert select_spill_victim(reqs, fast_pressure=True) == 2


def test_never_is_tabu_under_fast_pressure_too():
    # A fast request needs room and the ONLY resident session says 'never':
    # the fast request stays queued rather than evicting it.
    reqs = [_FakeReq(0, spill_class=SPILL_CLASS_NEVER), _FakeReq(7, fast=True)]
    assert select_spill_victim(reqs, fast_pressure=True) is None
    assert select_spill_victim(reqs, fast_pressure=False) is None


def test_all_never_yields_no_victim():
    reqs = [
        _FakeReq(0, spill_class=SPILL_CLASS_NEVER),
        _FakeReq(4, spill_class=SPILL_CLASS_NEVER),
    ]
    assert select_spill_victim(reqs, fast_pressure=True) is None


# -- "preferred" goes first ----------------------------------------------


def test_preferred_is_offered_before_fcfs():
    # The OLDEST session is 'preferred': FCFS alone would spill the youngest
    # normal (idx 2); the class overrides that and offers the preferred one.
    reqs = [
        _FakeReq(0, spill_class=SPILL_CLASS_PREFERRED),
        _FakeReq(4),
        _FakeReq(9),
    ]
    assert select_spill_victim(reqs) == 0
    assert select_spill_victim(reqs, fast_pressure=True) == 0


def test_preferred_does_not_shift_the_oldest_normal_tabu():
    """The oldest-untouchable tabu falls on the oldest NON-preferred session.

    Otherwise tagging a session 'preferred' would silently expose the oldest
    normal one as a victim."""
    reqs = [_FakeReq(0), _FakeReq(4, spill_class=SPILL_CLASS_PREFERRED)]
    # preferred is the victim; the oldest normal keeps its protection
    assert select_spill_victim(reqs) == 1
    # remove the preferred one -> the sole normal session is tabu again
    assert select_spill_victim([_FakeReq(0)]) is None


def test_sole_preferred_session_still_never_self_spills():
    # A lone session, even 'preferred', is not a victim under plain decode-OOM
    # (the sole-session rule is a physical one: spilling it relieves nothing).
    assert select_spill_victim([_FakeReq(3, spill_class=SPILL_CLASS_PREFERRED)]) is None


def test_preferred_ordering_within_class_stays_fcfs():
    reqs = [
        _FakeReq(1, spill_class=SPILL_CLASS_PREFERRED),
        _FakeReq(8, spill_class=SPILL_CLASS_PREFERRED),
        _FakeReq(4),
    ]
    # both preferred are below the normal one; youngest preferred goes first
    assert select_spill_victim(reqs) == 1
    keys = [session_priority_key(r) for r in reqs]
    assert keys[2] > keys[0] > keys[1]


def test_minimal_eviction_respects_the_class():
    """sizes/need pick the youngest SUFFICIENT candidate -- within the class
    order, so a preferred session that covers the need wins over a normal
    one that also would."""
    reqs = [
        _FakeReq(0),
        _FakeReq(4, spill_class=SPILL_CLASS_PREFERRED),
        _FakeReq(9),
    ]
    sizes = [100, 100, 100]
    assert select_spill_victim(reqs, sizes=sizes, need=50) == 1
    # preferred too small -> fall through to the FCFS-youngest sufficient one
    assert select_spill_victim(reqs, sizes=[100, 10, 100], need=50) == 2


def test_blocked_cooldown_composes_with_the_class():
    reqs = [
        _FakeReq(0),
        _FakeReq(4, spill_class=SPILL_CLASS_PREFERRED),
        _FakeReq(9),
    ]
    # cooldown excludes the preferred victim -> next in the class order
    assert select_spill_victim(reqs, blocked={1}) == 2
    assert select_spill_victim(reqs, blocked={1, 2}) is None  # only the tabu left


# -- server-arg gate ------------------------------------------------------


def _server_args(**kw):
    """``model_path='dummy'`` short-circuits ``__post_init__``, so the
    kv-session-offload handler can be driven in isolation (no GPU, no model)."""
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(model_path="dummy", **kw)


def test_default_spill_class_is_normal():
    assert _server_args().kv_session_offload_default_spill_class == SPILL_CLASS_NORMAL
    _server_args()._handle_kv_session_offload()  # must not raise


def test_non_default_class_requires_the_feature_flag():
    args = _server_args(kv_session_offload_default_spill_class=SPILL_CLASS_NEVER)
    with pytest.raises(ValueError, match="enable-kv-session-offload"):
        args._handle_kv_session_offload()


def test_unknown_default_class_is_rejected():
    args = _server_args(
        enable_kv_session_offload=True,
        kv_session_offload_default_spill_class="bogus",
    )
    with pytest.raises(ValueError, match="must be one of"):
        args._handle_kv_session_offload()


def test_valid_default_class_with_the_feature_passes():
    args = _server_args(
        enable_kv_session_offload=True,
        kv_session_offload_default_spill_class=SPILL_CLASS_PREFERRED,
    )
    args._handle_kv_session_offload()
    assert args.kv_session_offload_default_spill_class == SPILL_CLASS_PREFERRED


def test_server_arg_choices_match_the_canonical_list():
    """The CLI choices and the runtime constant list must not drift apart."""
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    actions = [
        a
        for a in parser._actions
        if "--kv-session-offload-default-spill-class" in a.option_strings
    ]
    assert len(actions) == 1
    assert sorted(actions[0].choices) == sorted(SPILL_CLASSES)
    assert actions[0].default == SPILL_CLASS_NORMAL


# -- request plumbing -----------------------------------------------------


def test_generate_req_input_carries_the_class_into_batch_items():
    from sglang.srt.managers.io_struct import GenerateReqInput

    obj = GenerateReqInput(text=["a", "b"], spill_class=SPILL_CLASS_NEVER)
    obj.normalize_batch_and_arguments()
    assert [obj[i].spill_class for i in range(2)] == [SPILL_CLASS_NEVER] * 2
    plain = GenerateReqInput(text=["a", "b"])
    plain.normalize_batch_and_arguments()
    assert plain[0].spill_class is None  # untouched until the tokenizer manager


def test_tokenizer_manager_applies_the_server_default():
    import types

    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    stub = types.SimpleNamespace(default_spill_class=SPILL_CLASS_PREFERRED)
    obj = GenerateReqInput(text="a")
    TokenizerManager._set_default_spill_class(stub, obj)
    assert obj.spill_class == SPILL_CLASS_PREFERRED
    # an explicit value always wins over the server default
    obj = GenerateReqInput(text="a", spill_class=SPILL_CLASS_NEVER)
    TokenizerManager._set_default_spill_class(stub, obj)
    assert obj.spill_class == SPILL_CLASS_NEVER


def test_tokenizer_manager_rejects_an_unknown_class():
    import types

    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    stub = types.SimpleNamespace(default_spill_class=SPILL_CLASS_NORMAL)
    obj = GenerateReqInput(text="a", spill_class="latency-critical")
    with pytest.raises(ValueError, match="spill_class must be one of"):
        TokenizerManager._set_default_spill_class(stub, obj)


def test_req_defaults_to_normal():
    """A Req built by any internal path (no spill_class in sight) must land in
    the NORMAL class -- that is what keeps the stock FCFS order intact."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = Req(
        rid="r0",
        origin_input_text="a",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )
    assert req.spill_class == SPILL_CLASS_NORMAL
    assert spill_class_of(req) == SPILL_CLASS_NORMAL
