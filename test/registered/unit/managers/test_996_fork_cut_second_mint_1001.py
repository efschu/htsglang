"""#996 SECOND CHUNKED MINT IN ONE PASS -- reached through a P-FORK-CUT chunk.

WHAT DIED. NF y6s (50fa5c42d1), P log ...dauer10012224_50fa5c42d1_1001_222413,
22:27:29, all three P ranks, on the first agent wave:

  P-FORK-CUT CUT  rid=pdflip-0-1 site=add_one_req prefix=8704 extend=16384
                  prompt=28725 fork=16640 src=told cut=16640 chunk=16384
  P-FORK-CUT SKIP rid=pdflip-0-3 site=add_one_req prefix=8640 extend=8448
                  prompt=63216 fork=8704 ... paid
  AssertionError: #996 SECOND CHUNKED MINT IN ONE PASS at add_one_req: this
                  adder already minted rid=pdflip-0-1 and is now being asked to
                  mint rid=pdflip-0-3

THE GAP. The told fork cut pdflip-0-1's chunk to 7936 of the 16384 chunk
budget; 8448 stayed in ``rem_chunk_tokens`` and the admission loop went on.
pdflip-0-3 entered the same truncating branch of ``add_one_req``, whose #959
guard asked only for the RESIDENT continuation (``chunked_req_outstanding``),
not for this pass's own mint (``new_chunked_req``) -- the end-anchor branch
asks for both. Pre-existing since P-FORK-CUT (28./29.09.); the PLE hint fix
in the same image changes nothing the adder reads.

THE FIX keeps the assert and refuses the second FRESH request one pass, as
the resident case does. The adder is the real ``PrefillAdder``; the cut is the
real ``p_fork_cut.apply`` on the told fork, with the metal geometry (page 64,
chunk 16384, 8448 left behind).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from flliper.srt import runtime_context as rc
from flliper.srt.managers import schedule_policy as sp
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from flliper.srt.utils.common import Range
from flliper.srt.pdflip import p_fork_cut as pfc
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15)

PAGE = 64
CHUNK = 16384
AVAIL = 1_000_000


def _tree_cache():
    tc = MagicMock()
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.evictable_size.return_value = 0
    tc.disable = False
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    return tc


def _allocator():
    al = MagicMock()
    al.full_available_size.return_value = AVAIL
    al.swa_available_size.return_value = AVAIL
    al.available_size.return_value = AVAIL
    return al


def _fresh_req(rid, n_tokens, fork=0):
    r = MagicMock(spec=Req)
    r.rid = str(rid)
    r.priority = 0
    r.prefix_indices = []
    r.full_untruncated_fill_ids = list(range(n_tokens))
    r.origin_input_ids = list(range(n_tokens))
    r.output_ids = []
    r.host_hit_length = 0
    r.swa_host_hit_length = 0
    r.sampling_params = SimpleNamespace(max_new_tokens=8, ignore_eos=False)
    r.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    r.retracted_stain = False
    r.born_spilled = False
    r.born_spilled_deep = False
    r.last_node = None
    r.mamba_pool_idx = None
    r._pdflip_fork_told = int(fork)  # PP0's verdict off the told (#1400/#1416)
    r.finished.return_value = False
    r.needs_host_load_back.return_value = False
    r.set_extend_range = MagicMock(
        side_effect=lambda start, end: setattr(r, "extend_range", Range(start, end))
    )
    return r


class _GroupP:
    """The real server args, seen as NF's P form (PP3, one attention rank)."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        over = {"pp_size": 3, "tp_size": 1, "chunked_prefill_size": CHUNK}
        if name in over:
            return over[name]
        return getattr(self._base, name)


@pytest.fixture
def adder(monkeypatch):
    sa = ServerArgs(model_path="dummy")
    set_global_server_args_for_scheduler(sa)
    monkeypatch.setattr(rc, "get_server_args", lambda: _GroupP(sa))
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.delenv("FLLIPER_PDFLIP_MAMBA_ANCHOR_INTERVAL", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_FORM", raising=False)
    # schedule_policy reads the group ONCE per process (_pdflip_chunk_admit /
    # _pdflip_park_on cache their verdict in module globals).  With
    # FLLIPER_PDFLIP_GROUP=P that verdict is True and it outlives this test's
    # monkeypatch: every later file in the same pytest process that builds a
    # real PrefillAdder then prices the next CHUNK instead of the whole extend,
    # the NO_TOKEN gate falls away and the add is refused OTHER (nb14:
    # test_nf_form_a_admission_follow_h105 4 red in a chunk run, 13/13 green
    # alone).  Plain assignment, not monkeypatch: the undo of a monkeypatch
    # would put back whatever a polluter left before this test.
    sp._PDFLIP_CHUNK_ADMIT = None
    sp._PDFLIP_PARK_ON = None
    sp._SECOND_CONTINUATION_REFUSALS.clear()
    running = MagicMock()
    running.reqs = []
    a = PrefillAdder(
        page_size=PAGE,
        tree_cache=_tree_cache(),
        token_to_kv_pool_allocator=_allocator(),
        running_batch=running,
        new_token_ratio=1.0,
        rem_input_tokens=1_000_000,
        rem_chunk_tokens=CHUNK,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )
    yield a
    sp._SECOND_CONTINUATION_REFUSALS.clear()
    sp._PDFLIP_CHUNK_ADMIT = None  # see above: never leave the P verdict behind
    sp._PDFLIP_PARK_ON = None


def _admit(adder, req):
    return adder.add_one_req(req, truncation_align_size=None)


def _cut_first(adder):
    """pdflip-0-1's shape: a free told-fork cut that leaves chunk budget behind."""
    assert pfc.armed(), "the P-FORK-CUT must be armed, or nothing below is reached"
    first = _fresh_req("pdflip-0-1", 20000, fork=7990)
    _admit(adder, first)
    assert adder.new_chunked_req is first
    assert first.extend_range.end - first.extend_range.start == 7936, (
        "the told fork must cut the chunk (7936 = floor_64(7990)), or the "
        "budget is spent and the second mint is unreachable"
    )
    assert adder.rem_chunk_tokens == CHUNK - 7936 == 8448, "metal: 8448 left behind"
    return first


def test_second_truncating_request_after_a_fork_cut_is_refused_not_minted(adder):
    """RED on 50fa5c42d1: the #996 assert fires at add_one_req."""
    first = _cut_first(adder)
    second = _fresh_req("pdflip-0-3", 63216)
    res = _admit(adder, second)
    assert res == AddReqResult.OTHER
    assert adder.new_chunked_req is first, "the first mint keeps the single field"
    assert second not in adder.can_run_list, "a refused request must not be in the batch"
    assert sp._SECOND_CONTINUATION_REFUSALS.get("add_one_req") == 1, "refusal is counted (#967)"
    # the first request's committed geometry is untouched (no re-prefill)
    assert (first.extend_range.start, first.extend_range.end) == (0, 7936)


def test_ignore_eos_site_refuses_the_same_second_mint(adder):
    """The sibling fresh-request mint site takes ALL of the left budget, so the
    same pass-local mint reaches it too (RED on 50fa5c42d1)."""
    first = _cut_first(adder)
    second = _fresh_req("eos", 63216)
    second.sampling_params = SimpleNamespace(max_new_tokens=8, ignore_eos=True)
    res = adder.add_one_req_ignore_eos(second)
    assert res == AddReqResult.OTHER
    assert adder.new_chunked_req is first
    assert second not in adder.can_run_list
    assert sp._SECOND_CONTINUATION_REFUSALS.get("add_one_req_ignore_eos") == 1


def test_whole_fit_request_still_rides_the_left_budget(adder):
    """The other direction: the refusal costs no throughput where no second
    continuation is involved -- a prompt that fits the leftover still joins."""
    first = _cut_first(adder)
    small = _fresh_req("small", 2000)
    _admit(adder, small)
    assert small in adder.can_run_list
    assert adder.new_chunked_req is first
    assert sp._SECOND_CONTINUATION_REFUSALS.get("add_one_req") is None


def test_without_a_mint_the_truncating_request_is_still_chunked(adder):
    """No first mint in this pass -> the ordinary chunked admission happens."""
    second = _fresh_req("pdflip-0-3", 63216)
    _admit(adder, second)
    assert adder.new_chunked_req is second
    assert second.extend_range.end - second.extend_range.start == CHUNK
