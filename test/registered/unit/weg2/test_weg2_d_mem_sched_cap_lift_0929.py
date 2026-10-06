"""NF1d 09291811 (z30y3f @ e952a6b709): D died in the first bs2 cell.

THE METAL. bs1 2k..245k ran; the 245k request ended at S7 and the end event
left a pending shrink to S0 (18:21:35 ``cap=32768 pending S-->S0 stage=S7
floor=248832``: the retained tree holds pages up to 248832). The first bs2
cell's 2113 tokens fit S0, so nothing cancelled the pending shrink and the
cap stayed at 32768. Every id below it was the tree's, every leaf the peel
freed sat above it: "EVICTION UNDER-DELIVERED: asked for 2177 tokens, the
pool received 0 ... A RESIDENCY CAP IS ENGAGED and is holding 278208 slot
ids" -> ``RuntimeError: Prefill out of memory`` on TP0/TP1/TP2 at once, the
abort's GPU coredumps, W17_Weg2GroupDead.

WHAT MUST HOLD. While a shrink is pending, the group's demand is paid: if the
free ids below the pending cap cannot pay it on every rank (replicated MIN),
the cap goes back to the MAPPED stage; with no demand left it returns to the
pending cap. Room below the cap keeps the design (new pages below).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from test_weg2_d_mem_sched_0929 import FLOOR_LADDER, _req, floor_env, tick_env  # noqa: E402,F401


def _alloc(free_ids):
    return types.SimpleNamespace(free_pages=torch.tensor(free_ids, dtype=torch.int64),
                                 release_pages=torch.empty(0, dtype=torch.int64))


def _pending_s0(dsv, sched, caps, floor):
    """weg2-2-10 at S1 finishes; its retained pages hold the shrink: pending S0."""
    sched.running_batch.reqs = [_req("weg2-2-10", 32835, 258)]
    dsv.runtime_tick(sched)
    sched.running_batch.reqs = []
    floor["page"] = 517
    st = dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    assert not st.changed and ms.pending == 0 and caps[-1] == FLOOR_LADDER[0]
    return ms


def test_a_pending_cap_without_room_below_hands_the_admission_the_mapped_stage(floor_env):
    dsv, sched, caps, floor = floor_env
    # page 64: the cap of 32768 is page 512; the tree holds every id below it
    sched.token_to_kv_pool_allocator = _alloc(list(range(600, 1024)))
    ms = _pending_s0(dsv, sched, caps, floor)
    sched.waiting_queue = [_req("bs2-a", 2113)]          # fits S0: cancels nothing
    st = dsv.runtime_tick(sched)
    assert not st.changed and ms.pending == 0 and ms.stage == 1
    assert caps[-1] == FLOOR_LADDER[1], (
        "the pending cap %d stayed engaged with no free id below it: the 2113-token "
        "admission cannot be paid (NF1d Prefill out of memory)" % caps[-1])
    # the request runs: still no room below -> the stage keeps paying
    sched.waiting_queue = []
    sched.running_batch.reqs = [_req("bs2-a", 2113, 10)]
    n = len(caps)
    dsv.runtime_tick(sched)
    assert len(caps) == n and caps[-1] == FLOOR_LADDER[1]
    # the group is idle again: back to the pending cap
    sched.running_batch.reqs = []
    dsv.runtime_tick(sched)
    assert caps[-1] == FLOOR_LADDER[0] and ms.pending == 0


def test_room_below_the_pending_cap_keeps_it(floor_env):
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 512)))  # 412 pages below
    ms = _pending_s0(dsv, sched, caps, floor)
    n = len(caps)
    sched.waiting_queue = [_req("bs2-a", 2113)]
    dsv.runtime_tick(sched)
    assert len(caps) == n and caps[-1] == FLOOR_LADDER[0] and ms.pending == 0


def test_one_rank_without_room_lifts_the_cap_on_every_rank(floor_env):
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 512)))  # room HERE
    sched._weg2_group_min_ints = lambda vals: [min(list(vals) + [0])]  # a peer has none
    _pending_s0(dsv, sched, caps, floor)
    sched.waiting_queue = [_req("bs2-a", 2113)]
    dsv.runtime_tick(sched)
    assert caps[-1] == FLOOR_LADDER[1]


def test_the_lift_writes_its_line(floor_env, caplog):
    import logging

    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(600, 1024)))
    _pending_s0(dsv, sched, caps, floor)
    caplog.set_level(logging.INFO)
    sched.waiting_queue = [_req("bs2-a", 2113)]
    dsv.runtime_tick(sched)
    lines = [m for m in caplog.messages if dsv.CAP_LIFT_MARK in m]
    assert lines and "lifted=yes" in lines[-1] and "cap=%d" % FLOOR_LADDER[1] in lines[-1]


def test_free_tokens_below_counts_only_ids_under_the_cap():
    from sglang.srt.weg2 import d_seat_vram as dsv

    a = _alloc([1, 2, 511, 512, 513, 900])
    assert dsv.free_tokens_below(a, 32768, 64) == 4 * 64
    assert dsv.free_tokens_below(object(), 32768, 64) is None


# ---------------------------------------------------------------------------
# NF 1528 (K2 54f8fb3413, 05.10. 20:49:25Z): the lift paid for chunk 1 only
# ---------------------------------------------------------------------------
# weg2-54-290 (3006 tokens, chunks 2176 + 830): the tick before chunk 1 saw
# incoming=3006 -> lift (need 3006). After chunk 1 the request is the
# ``chunked_req``: ``used`` 3006, ``incoming`` 0, need 64 -> the 192 free ids
# under the pending cap "paid" it, the cap fell back to 163840 and chunk 2
# (830 tokens) found 192 free: "Prefill out of memory" on TP0/TP1/TP2.


def _chunked(rid, n_in, done):
    """A chunked_req that has prefilled ``done`` of ``n_in`` tokens."""
    r = _req(rid, n_in)
    r.prefix_indices = [0] * done
    r.extend_range = types.SimpleNamespace(start=0, end=done)
    return r


def _caplift_lines(caplog, dsv):
    return [m for m in caplog.messages if dsv.CAP_LIFT_MARK in m]


def test_b_the_next_chunk_of_a_live_chunked_req_is_demand(floor_env, caplog):
    import logging

    dsv, sched, caps, floor = floor_env
    # page 64, cap 32768 = page 512: 3 free ids below it (192 tokens), the rest above
    sched.token_to_kv_pool_allocator = _alloc([100, 101, 102, 700, 701, 702])
    ms = _pending_s0(dsv, sched, caps, floor)
    caplog.set_level(logging.INFO)
    sched.chunked_req = _chunked("weg2-54-290", 3006, 2176)  # chunk 2 = 830 tokens
    sched.waiting_queue = []
    dsv.runtime_tick(sched)
    lines = _caplift_lines(caplog, dsv)
    assert ms._cap_lifted is True, (
        "the pending cap fell back with 830 tokens of chunk 2 to come and 192 free ids "
        "below it (NF 1528: Prefill out of memory)")
    assert caps[-1] == FLOOR_LADDER[1]
    assert lines and "lifted=yes" in lines[-1]
    assert "need=%d" % (830 + 64) in lines[-1] and "rest=830" in lines[-1]
    assert "room=192" in lines[-1]


def test_c_a_two_chunk_request_whose_rest_fits_keeps_the_pending_cap(floor_env, caplog):
    import logging

    dsv, sched, caps, floor = floor_env
    # weg2-32-205: chunk 2 of 428 tokens (+ one page for the one seat = 492) fits the
    # 8 free ids (512 tokens) below the cap
    sched.token_to_kv_pool_allocator = _alloc([100, 101, 102, 103, 104, 105, 106, 107, 700])
    ms = _pending_s0(dsv, sched, caps, floor)
    caplog.set_level(logging.INFO)
    n = len(caps)
    sched.chunked_req = _chunked("weg2-32-205", 3052, 2624)
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is False and len(caps) == n and caps[-1] == FLOOR_LADDER[0]
    assert not _caplift_lines(caplog, dsv)


def test_d_the_lift_ends_after_the_last_chunk(floor_env, caplog):
    import logging

    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc([100, 101, 102, 700, 701, 702])
    ms = _pending_s0(dsv, sched, caps, floor)
    sched.chunked_req = _chunked("weg2-54-290", 3006, 2176)
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is True and caps[-1] == FLOOR_LADDER[1]
    caplog.set_level(logging.INFO)
    # the last chunk ran: the request is gone from chunked_req and the batch
    sched.chunked_req = None
    sched.running_batch.reqs = []
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is False and caps[-1] == FLOOR_LADDER[0] and ms.pending == 0
    lines = _caplift_lines(caplog, dsv)
    assert lines and "used=0 incoming=0 rest=0 need=0" in lines[-1] and "lifted=no" in lines[-1]


def test_e_only_the_next_chunk_counts_not_the_whole_rest(floor_env, caplog):
    import logging

    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 160)))  # 3840 tokens below
    ms = _pending_s0(dsv, sched, caps, floor)
    caplog.set_level(logging.INFO)
    sched.chunked_req = _chunked("long", 20000, 4096)   # rest 15904, chunk 4096
    dsv.runtime_tick(sched)
    # need = min(15904, 4096) + 64 = 4160 > 3840 -> lift; the whole rest would also lift
    assert ms._cap_lifted is True
    assert "rest=15904 need=%d" % (4096 + 64) in _caplift_lines(caplog, dsv)[-1]


def test_f_chunked_rest_is_the_overestimate(floor_env):
    dsv, sched, caps, floor = floor_env
    assert dsv._chunked_rest(sched) == 0
    sched.chunked_req = _chunked("p", 3006, 2176)
    assert dsv._chunked_rest(sched) == 830
    # a parked chunk: the fill boundary ran ahead of the prefix -> the smaller counts
    sched.chunked_req.extend_range = types.SimpleNamespace(start=0, end=3006)
    assert dsv._chunked_rest(sched) == 830
    sched.chunked_req.prefix_indices = [0] * 3006
    assert dsv._chunked_rest(sched) == 0
    # a desk double with neither field: nothing is known as done -> the whole request
    sched.chunked_req = _req("q", 100)
    assert dsv._chunked_rest(sched) == 100


def test_g_a_cached_verdict_does_not_outlive_a_chunk(floor_env, monkeypatch):
    """With the re-check on (K>1) a decode-only round reuses the last group
    verdict. A live chunked_req is not decode-only: chunk 2 of a 3-chunk
    request eats the room chunk 1 left, while ``need`` (one chunk + a page)
    stays the same -- the cache key alone would still say "room"."""
    dsv, sched, caps, floor = floor_env
    monkeypatch.setenv("SGLANG_WEG2_D_MEM_RECHECK_ROUNDS", "8")
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 200)))   # 6464 tokens below
    ms = _pending_s0(dsv, sched, caps, floor)
    sched.chunked_req = _chunked("long", 20000, 4096)                  # rest 15904: need 4160
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is False                                     # the room pays chunk 2
    dsv.runtime_tick(sched)                                            # 2nd read: the back-off grows
    assert ms._cap_lifted is False
    sched.token_to_kv_pool_allocator = _alloc([100, 101, 102])         # chunk 2 took it
    sched.chunked_req = _chunked("long", 20000, 8192)                  # rest 11808: need 4160 again
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is True, "the cached 'room' verdict was reused across a chunk"


@pytest.mark.parametrize("group,stage_tokens", [("", None), ("P", None), ("D", None), ("D", "262144")])
def test_h_flip_unchanged_a_live_chunked_req_adds_nothing_outside_the_stage_form(
        monkeypatch, group, stage_tokens):
    """The 27B flip / P / a D without stage form leave the tick before the
    cap-lift block: a chunked_req there costs no collective, no sync, no
    allocation and builds no machine (the 1528 rest is read only inside)."""
    from test_weg2_d_mem_sched_0929 import _no_side_effects

    from sglang.srt.weg2 import d_seat_vram as dsv

    monkeypatch.setenv("SGLANG_WEG2_GROUP", group)
    monkeypatch.setenv("SGLANG_OPT_WEG2_D_SEAT_VRAM", "1")
    if stage_tokens is None:
        monkeypatch.delenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", stage_tokens)
    hits, sched = _no_side_effects(monkeypatch, dsv)
    sched.chunked_req = _chunked("weg2-54-290", 3006, 2176)
    setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e", n=6, cap=6, done=True))
    calls = []
    monkeypatch.setattr(dsv, "_chunked_rest", lambda s: calls.append(1) or 0)
    for _ in range(3):
        assert dsv.runtime_tick(sched) is None
    assert hits == [] and calls == []
    assert not hasattr(sched, dsv.MEM_SCHED_ATTR)


# ---------------------------------------------------------------------------
# NF 1540 (cand3 bce16a6ddf, 05.10. 23:45:03Z): the lift did not count the load-back
# ---------------------------------------------------------------------------
# weg2-62-415: 27620 tokens, 26048 of them a HOST-backed prefix (``pp_load_back_extent``
# 26048), 1572 to extend. Pending S0 (cap 32768) with 26880 free tokens (420 ids) below
# it, 19200 evictable tokens above it. The tick weighed one chunk (4096 + 0 seats) against
# 26880 -> no lift; ``init_load_back`` put the 26048 on the device in one step, 832 were
# left for the 1572 extend ("Prefill out of memory", EVICTION UNDER-DELIVERED 64 of 868).


def _hosted(rid, n_in, load_back):
    r = _req(rid, n_in)
    r.pp_load_back_extent = load_back
    return r


def test_i_a_host_load_back_is_paid_whole_by_the_lift(floor_env, caplog):
    import logging

    dsv, sched, caps, floor = floor_env
    # page 64, cap 32768 = page 512: 420 free ids below it = 26880 tokens, the rest above
    sched.token_to_kv_pool_allocator = _alloc(list(range(10, 430)) + list(range(600, 700)))
    ms = _pending_s0(dsv, sched, caps, floor)
    caplog.set_level(logging.INFO)
    sched.waiting_queue = [_hosted("weg2-62-415", 27620, 26048)]
    dsv.runtime_tick(sched)
    assert ms.pending == 0, "the machine itself keeps the pending shrink (27620 fits S0)"
    lines = _caplift_lines(caplog, dsv)
    assert ms._cap_lifted is True and caps[-1] == FLOOR_LADDER[1], (
        "the pending cap stayed at 32768: the 26048-token load-back + 1572 extend need "
        "27620 and only 26880 ids are free below it (NF 1540: Prefill out of memory)")
    assert lines and "lifted=yes" in lines[-1]
    assert "need=27620" in lines[-1] and "room=26880" in lines[-1]


def test_j_a_load_back_the_room_pays_keeps_the_pending_cap(floor_env, caplog):
    import logging

    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(10, 450)) + list(range(600, 700)))  # 28160
    ms = _pending_s0(dsv, sched, caps, floor)
    caplog.set_level(logging.INFO)
    n = len(caps)
    sched.waiting_queue = [_hosted("weg2-62-415", 27620, 26048)]
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is False and len(caps) == n and caps[-1] == FLOOR_LADDER[0]
    assert not _caplift_lines(caplog, dsv)


def test_k_without_a_load_back_extent_the_need_is_one_chunk(floor_env):
    """The request has not been matched yet (no extent) or has no host hit: the
    need stays the first chunk, byte-for-byte what 1528 computed."""
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(10, 430)) + list(range(600, 700)))
    ms = _pending_s0(dsv, sched, caps, floor)
    for ext in (None, 0):
        n = len(caps)
        sched.waiting_queue = [_hosted("weg2-62-415", 27620, ext)]
        dsv.runtime_tick(sched)
        assert ms._cap_lifted is False and len(caps) == n and caps[-1] == FLOOR_LADDER[0]
    plain = _req("weg2-62-416", 27620)          # a double without the field at all
    sched.waiting_queue = [plain]
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is False and caps[-1] == FLOOR_LADDER[0]


def test_l_queued_load_backs_add_up_and_stay_inside_their_prompt():
    from sglang.srt.weg2 import d_seat_vram as dsv

    assert dsv._queued_load_back([]) == 0
    assert dsv._queued_load_back([_req("a", 100), _hosted("b", 100, None)]) == 0
    assert dsv._queued_load_back([_hosted("a", 14000, 13000), _hosted("b", 14000, 13000)]) == 26000
    assert dsv._queued_load_back([_hosted("a", 1000, 5000)]) == 1000      # clamped to its own prompt
    assert dsv._queued_load_back([_hosted("a", 1000, -5)]) == 0


def test_m_two_queued_load_backs_are_paid_together(floor_env):
    dsv, sched, caps, floor = floor_env
    sched.token_to_kv_pool_allocator = _alloc(list(range(10, 430)) + list(range(600, 700)))  # 26880
    ms = _pending_s0(dsv, sched, caps, floor)
    sched.waiting_queue = [_hosted("a", 14000, 13000)]       # one alone fits: 13000 + 1000
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is False
    sched.waiting_queue = [_hosted("a", 14000, 13000), _hosted("b", 14000, 13000)]
    dsv.runtime_tick(sched)                                  # 26000 + 2000 > 26880
    assert ms._cap_lifted is True and caps[-1] == FLOOR_LADDER[1]


@pytest.mark.parametrize("group,stage_tokens", [("", None), ("P", None), ("D", None), ("D", "262144")])
def test_n_flip_unchanged_a_queued_load_back_adds_nothing_outside_the_stage_form(
        monkeypatch, group, stage_tokens):
    """The 27B flip / P / a D without stage form leave the tick before the
    cap-lift block: a queued load-back is never read there."""
    from test_weg2_d_mem_sched_0929 import _no_side_effects

    from sglang.srt.weg2 import d_seat_vram as dsv

    monkeypatch.setenv("SGLANG_WEG2_GROUP", group)
    monkeypatch.setenv("SGLANG_OPT_WEG2_D_SEAT_VRAM", "1")
    if stage_tokens is None:
        monkeypatch.delenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", stage_tokens)
    hits, sched = _no_side_effects(monkeypatch, dsv)
    sched.waiting_queue = [_hosted("weg2-62-415", 27620, 26048)]
    setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e", n=6, cap=6, done=True))
    calls = []
    monkeypatch.setattr(dsv, "_queued_load_back", lambda adm: calls.append(1) or 0)
    for _ in range(3):
        assert dsv.runtime_tick(sched) is None
    assert hits == [] and calls == []
    assert not hasattr(sched, dsv.MEM_SCHED_ATTR)


# ---------------------------------------------------------------------------
# NF 4p5am6 (06.10. 22:10:12Z, weg2-40-295): "lifted=no bei rest+64>room" between chunks
# ---------------------------------------------------------------------------
# D log of boot dkrnfint4h6ablxcbar1dauer10062134_50632b25e1 (TP0/1/2 alike):
#   CAP-LIFT pending=S3 stage=S4 used=73158 incoming=0 rest=9982 need=4224 room=5120 lifted=no
# The evaluation rule read it as a violation (rest 9982 + 64 > room 5120). It is not: the
# lift prices the NEXT chunk (``min(rest, chunked_prefill_size)`` = 4096) + one page per
# seat (2 x 64), i.e. need 4224 <= room 5120 -- lifted=no is the verdict of the code, and
# ``rest`` is only the (over-estimated, tick-before-stash) tail of the whole request.
# The real gap between two chunks is another one (1534 F6, tests q..s below): a
# D-SEAT-REWAKE GROW/SHRINK iteration skipped the whole tick, so the room that chunk 2
# ate was never weighed against chunk 3 before admission.


def test_o_the_4p5am6_line_prices_one_chunk_not_the_whole_rest(floor_env, caplog):
    import logging

    dsv, sched, caps, floor = floor_env
    # 80 free ids below the cap = 5120 tokens: first a lift (1 id), then the room of the log
    sched.token_to_kv_pool_allocator = _alloc([100])
    ms = _pending_s0(dsv, sched, caps, floor)
    caplog.set_level(logging.INFO)
    sched.chunked_req = _chunked("weg2-40-295", 12000, 12000 - 9982)
    sched.running_batch.reqs = [_req("weg2-40-294", 1000, 100)]   # the second seat
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is True
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 180)))   # 5120 tokens below
    dsv.runtime_tick(sched)
    line = _caplift_lines(caplog, dsv)[-1]
    assert ms._cap_lifted is False and "lifted=no" in line
    assert "rest=9982 need=4224 room=5120" in line, line
    # the invariant an evaluation may check: lifted=no <=> the priced need fits the room
    need, room = (int(line.split(k + "=")[1].split()[0]) for k in ("need", "room"))
    assert need <= room < 9982 + 64


def _seat_move(monkeypatch, moved="grow"):
    """The iteration of a D-SEAT-REWAKE GROW/SHRINK: ``tick`` moved n, so
    ``round_boundary`` does not run the stage tick (``V.runtime_tick``)."""
    from sglang.srt.weg2 import d_seat_rewake as rw

    monkeypatch.setattr(rw, "tick", lambda sched: moved)
    return rw


def test_q_a_seat_move_iteration_weighs_the_next_chunk_against_the_pending_cap(
        floor_env, caplog, monkeypatch):
    """1534 F6 on the 4p5am6 shape: chunk 2 took the room, chunk 3 (4096 + 2 seats)
    needs more than is left below the pending cap, and the iteration between them
    is a GROW (22:10:15 TP0 ``D-SEAT-REWAKE GROW n=2->3``) that skipped the tick."""
    import logging

    dsv, sched, caps, floor = floor_env
    rw = _seat_move(monkeypatch)
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 180)))   # 5120 below: pays 4224
    ms = _pending_s0(dsv, sched, caps, floor)
    sched.chunked_req = _chunked("weg2-40-295", 12000, 12000 - 9982)
    sched.running_batch.reqs = [_req("weg2-40-294", 1000, 100)]
    dsv.runtime_tick(sched)
    assert ms._cap_lifted is False
    caplog.set_level(logging.INFO)
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 140)))   # chunk 2 ate 2560 of it
    sched.chunked_req = _chunked("weg2-40-295", 12000, 12000 - 7358)
    assert rw.round_boundary(sched) == "grow"
    assert ms._cap_lifted is True, "the seat-move iteration let chunk 3 allocate under the pending cap"
    assert caps[-1] == FLOOR_LADDER[1]
    line = _caplift_lines(caplog, dsv)[-1]
    assert "lifted=yes" in line and "rest=7358" in line and "need=4224" in line and "room=2560" in line


def test_r_a_seat_move_iteration_runs_no_stage_step(floor_env, monkeypatch):
    """Only the lift verdict runs there: no ``MemSched.step`` (its round counter and
    floor reads stay), no ``apply_stage``, and a room that pays keeps the cap."""
    dsv, sched, caps, floor = floor_env
    rw = _seat_move(monkeypatch)
    sched.token_to_kv_pool_allocator = _alloc(list(range(100, 180)))
    ms = _pending_s0(dsv, sched, caps, floor)
    sched.chunked_req = _chunked("weg2-40-295", 12000, 12000 - 9982)
    sched.running_batch.reqs = [_req("weg2-40-294", 1000, 100)]
    steps = []
    real_step = ms.step
    monkeypatch.setattr(ms, "step", lambda *a, **k: steps.append(1) or real_step(*a, **k))
    fr = dsv._floor_room_state(ms)
    ticks, n = fr["ticks"], len(caps)
    assert rw.round_boundary(sched) == "grow"
    assert steps == [] and fr["ticks"] == ticks          # no stage step, no tick count
    assert ms._cap_lifted is False and len(caps) == n    # the room pays: the cap is untouched


@pytest.mark.parametrize("group,stage_tokens", [("", None), ("P", None), ("D", None), ("D", "262144")])
def test_s_flip_unchanged_a_seat_move_adds_nothing_outside_the_stage_form(
        monkeypatch, group, stage_tokens):
    from test_weg2_d_mem_sched_0929 import _no_side_effects

    from sglang.srt.weg2 import d_seat_vram as dsv

    monkeypatch.setenv("SGLANG_WEG2_GROUP", group)
    monkeypatch.setenv("SGLANG_OPT_WEG2_D_SEAT_VRAM", "1")
    if stage_tokens is None:
        monkeypatch.delenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", stage_tokens)
    hits, sched = _no_side_effects(monkeypatch, dsv)
    sched.chunked_req = _chunked("weg2-54-290", 3006, 2176)
    setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e", n=6, cap=6, done=True))
    rw = _seat_move(monkeypatch)
    calls = []
    monkeypatch.setattr(dsv, "_chunked_rest", lambda s: calls.append(1) or 0)
    for _ in range(3):
        assert rw.round_boundary(sched) == "grow"
    assert hits == [] and calls == []
    assert not hasattr(sched, dsv.MEM_SCHED_ATTR)
