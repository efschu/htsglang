# SPDX-License-Identifier: Apache-2.0
"""Q-360 (27B dual y8p, boot ...dual1mpsleepbar1fs10030828_bd2e3bc22d): the
fork-cut hand-back tail is D's own extend, not a W31 over X.

METAL. ``SGLANG_WEG2_FORK_ANCHOR_TOKEN=248045`` on both groups (new against the
W35-free dual boots 10020527 .. 10012335, whose P cut every leg 1 to N-1). P's
intake cut each leg 1 BEFORE the generation prompt (``P-TRIM-END-ANCHOR ... fork
rid=weg2-0-6 tokens=4061->4054``, weg2-0-1 25->18, weg2-0-3 12744->12739); D's
store read ended at the fork, the GROUP match was F, and the X gate priced the
generation-prompt tail against the dual layout's X=1:

  D  WEG2 X-GATE rid=weg2-0-6 uncached=7 X=1 replicated_term=group verdict=W31
  D  W50 Weg2TpPrefillExceeded ... extent after prefix matching is 7
  front  W50-REROUTE n=1 -> P leg 1 again (writes the same F) -> D W31 again
  front  W35 Weg2XReQueueLoop rid=weg2-0-6 (503 to the client)

and weg2-0-3 (a stream) RESUME-VIA-P every 30 s on ``uncached=5``.

THE TESTS run the real P intake (``p_trim_end_anchor.split_ids``) for the leg-1
cut and the real D X gate (``Scheduler._weg2_x_refuses``) with the group match
the store read landed (= F). Red on bd2e3bc22d (W31 on every pass -> the W35
path), green after: the gate admits a tail <= N - F (``WEG2 X-GATE FORK-TAIL``).
Danger directions, each pinned: a match SHORT of the fork still prices against
X (X is not raised); the switch off keeps the old W31; group P and a request
with output are no fork hand-backs; every term is replicated (two ranks with
different local trees reach the same verdict).
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.managers import scheduler as SC
from sglang.srt.managers import tp_head_congruence as thc
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.weg2 import p_trim_end_anchor as PT
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

IM_START = 248045
#: a 7-token generation prompt (weg2-0-6's tail), opened by <|im_start|>
GEN7 = [IM_START, 74455, 198, 248068, 198, 271, 248069]
GEN5 = [IM_START, 74455, 198, 248068, 198]

P_ENV = {"SGLANG_WEG2_GROUP": "P", "SGLANG_WEG2_P_TRIM_END_ANCHOR": "1",
         "SGLANG_WEG2_FORK_ANCHOR_TOKEN": str(IM_START)}
D_ENV = {"SGLANG_WEG2_GROUP": "D", "SGLANG_WEG2_DUAL_LAYOUT": "1",
         "SGLANG_WEG2_FORK_ANCHOR_TOKEN": str(IM_START)}
SWITCHES = ("SGLANG_WEG2_GROUP", "SGLANG_WEG2_P_TRIM_END_ANCHOR", "SGLANG_WEG2_FORK_ANCHOR_TOKEN",
            "SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_FORK_ANCHOR_MAX_TAIL")


def _env(monkeypatch, env):
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def _prompt(n: int, gen=GEN7):
    body = [(i % 50000) + 11 for i in range(n - len(gen))]   # never 248045
    return body + list(gen)


def _p_leg1_cut(monkeypatch, rid, ids) -> int:
    """Group P's real intake: how many ids P computes for this leg 1."""
    _env(monkeypatch, P_ENV)
    recv = SimpleNamespace(rid=rid, input_ids=list(ids), input_embeds=None, return_logprob=False,
                           sampling_params=SimpleNamespace(max_new_tokens=1), session_params=None,
                           session_id=None, mm_inputs=None)
    head, tail = PT.split_ids(recv)
    assert tail is not None and len(head) + len(tail) == len(ids)
    return len(head)


def _gate(x: int = 1, tp_size: int = 3):
    stub = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
        ps=SimpleNamespace(tp_size=tp_size),
        tree_cache=SimpleNamespace(cache_controller=SimpleNamespace(
            mem_pool_host=SimpleNamespace(size=10 ** 9))),
    )
    stub.weg2_uncached_extent = lambda req, head=None: Scheduler.weg2_uncached_extent(stub, req, head)
    stub._weg2_host_carry_tokens = lambda: Scheduler._weg2_host_carry_tokens(stub)
    return stub


def _d_req(rid, ids, local_prefix: int, output=()):
    return SimpleNamespace(rid=rid, origin_input_ids=list(ids), full_untruncated_fill_ids=list(ids),
                           output_ids=list(output), prefix_indices=list(range(local_prefix)),
                           host_hit_length=0, return_logprob=False, input_embeds=None,
                           session_id=None, multimodal_inputs=None)


def _head(rid, group_match: int):
    canonical = thc.canonical_head_rids([rid])
    return thc.build_uniform_head_inputs(
        canonical, thc.build_head_order_payload(canonical, {rid: group_match}), None, True)


def _d_refuses(monkeypatch, rid, ids, group_match, env=D_ENV, output=(), x=1) -> bool:
    _env(monkeypatch, env)
    return Scheduler._weg2_x_refuses(_gate(x), _d_req(rid, ids, group_match, output), _head(rid, group_match))


# -- the metal shapes: P's fork cut, then D's gate on what the read landed --------

@pytest.mark.parametrize("rid,n,gen", [("weg2-0-6", 4061, GEN7), ("weg2-0-1", 25, GEN7),
                                       ("weg2-0-3", 12744, GEN5)])
def test_a_fork_cut_handback_is_admitted_by_dual_d(monkeypatch, caplog, rid, n, gen):
    ids = _prompt(n, gen)
    f = _p_leg1_cut(monkeypatch, rid, ids)
    assert n - f == len(gen), "P's real intake cut the leg before the generation prompt"
    caplog.set_level(logging.INFO, logger=SC.logger.name)
    assert _d_refuses(monkeypatch, rid, ids, f) is False, (
        f"D refused the {n - f}-token generation-prompt tail after a full P prefill -- "
        "the W50 -> re-leg -> W50 -> W35 path of boot 10030828")
    text = caplog.text
    assert f"WEG2 X-GATE FORK-TAIL rid={rid} uncached={n - f} tail={n - f} X=1 verdict=admit" in text
    assert f"WEG2 X-GATE rid={rid} uncached={n - f} X=1 replicated_term=group verdict=admit" in text


def test_the_w35_path_two_p_legs_never_reach_a_second_refusal(monkeypatch):
    """The front's W35 is a SECOND D refusal after a full P prefill. Each P leg of
    the rid writes the same fork cut, so D's verdict is the same on both passes:
    parent = W31, W31 (W35); fixed = admit on the first."""
    rid, ids = "weg2-0-6", _prompt(4061)
    refusals = 0
    for _leg in range(2):
        f = _p_leg1_cut(monkeypatch, rid, ids)
        if not _d_refuses(monkeypatch, rid, ids, f):
            break
        refusals += 1
    assert refusals == 0, f"D refused {refusals}x after P legs (2 = W35 Weg2XReQueueLoop)"


# -- danger directions --------------------------------------------------------------

def test_a_match_short_of_the_fork_still_prices_against_x(monkeypatch):
    """Not X raised: P's write not landed (match 0) or one token short of F stays W31."""
    rid, ids = "weg2-0-6", _prompt(4061)
    f = _p_leg1_cut(monkeypatch, rid, ids)
    assert _d_refuses(monkeypatch, rid, ids, 0) is True
    assert _d_refuses(monkeypatch, rid, ids, f - 1) is True
    assert _d_refuses(monkeypatch, rid, ids, f) is False


def test_switch_off_keeps_the_old_w31(monkeypatch):
    rid, ids = "weg2-0-6", _prompt(4061)
    off = {"SGLANG_WEG2_GROUP": "D", "SGLANG_WEG2_DUAL_LAYOUT": "1"}
    assert _d_refuses(monkeypatch, rid, ids, 4054, env=off) is True
    assert _d_refuses(monkeypatch, rid, ids, 4060, env=off) is False, "the N-1 hand-back as before"


def test_no_fork_handback_on_group_p_or_with_output(monkeypatch):
    rid, ids = "weg2-0-6", _prompt(4061)
    _env(monkeypatch, P_ENV)
    assert SC._weg2_fork_handback_tail(_d_req(rid, ids, 4054)) == 0, "group P cuts, it never consumes"
    assert _d_refuses(monkeypatch, rid, ids, 4054, output=[11, 12]) is True, \
        "a request that already generated is no fresh hand-back"
    _env(monkeypatch, D_ENV)
    assert SC._weg2_fork_handback_tail(_d_req("not-front", ids, 4054)) == 0, "only front rids"
    assert SC._weg2_fork_handback_tail(_d_req(rid, ids, 4054)) == 7


def test_the_verdict_is_the_groups_whatever_the_local_tree(monkeypatch):
    """Two ranks, different local trees (0 and 4060), one group match F: one verdict."""
    rid, ids = "weg2-0-6", _prompt(4061)
    _env(monkeypatch, D_ENV)
    head = _head(rid, 4054)
    a = Scheduler._weg2_x_refuses(_gate(), _d_req(rid, ids, 0), head)
    b = Scheduler._weg2_x_refuses(_gate(), _d_req(rid, ids, 4060), head)
    assert a is b is False


def test_the_allowance_is_bounded_by_the_fork_window(monkeypatch):
    """No fork token within max_tail: no allowance, the tail prices against X."""
    rid = "weg2-0-9"
    ids = _prompt(4061, gen=[IM_START] + [198] * 20)   # <|im_start|> 21 from the end
    _env(monkeypatch, D_ENV)
    assert SC._weg2_fork_handback_tail(_d_req(rid, ids, 0)) == 0
    assert _d_refuses(monkeypatch, rid, ids, 4061 - 21) is True
