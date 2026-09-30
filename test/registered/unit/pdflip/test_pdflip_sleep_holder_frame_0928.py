"""hc_combine holder at P's sleep (NF rc12z14 dkrnfh91dprsavisnoadoptbar1dauer09280956, 10:02:57Z /
10:03:54Z, PP0 and PP1, sleep=2 and sleep=3): the last forward's stage output (the hyper-connection
residual hc_combine writes, 3.4-4.2 MiB there, 56-188 MiB after a large last batch in the 08:31 boot)
is live across the sleep, held by ``dict['hidden_states'] <- PPProxyTensors.tensors`` -- the
GenerationBatchResult of the last forward, bound to the loop-local ``result`` of the PP event loop.
``release_memory_occupation`` runs INSIDE that loop (``_pp_forward_and_process_input_requests``), so
the frame keeps the previous pass's result for the whole sleep. The instrument could not say so: it
filtered frames as noise, and its own temporary lists crowded the four holder slots
('list <- list <- list <- ?'); on PP2 it died on a dead weakproxy in ``gc.get_objects()``.

Fix: (1) every PP loop drops ``result`` right after its last read (the proxy send); the transport keeps
what it still sends (P2PWork.payload / the torch Work). (2) the holder walk names frame locals, skips
its own temporaries, and survives dead weak proxies.
"""

import ast
import gc
import inspect
import textwrap
import weakref

import pytest

from flliper.srt.pdflip import sleep_staging as ss


class _Holder:
    def __init__(self, v):
        self.attr = v


class _Payload:
    pass


def test_a_frame_local_is_named_as_the_holder():
    """The real shape: the sleep RPC is served from INSIDE the loop whose local holds the
    result, i.e. the holder is a frame further up the live stack (no gc object in 3.11+)."""
    obj = _Payload()

    def event_loop():
        result = obj  # noqa: F841 -- the frame local under test
        return serve_sleep()

    def serve_sleep():
        return ss.describe_holders([obj], depth=3)

    del_obj_holders = event_loop()
    assert any(h == "frame event_loop local result" for h in del_obj_holders), del_obj_holders


def test_the_walk_does_not_report_its_own_temporaries():
    obj = _Payload()
    h = _Holder(obj)  # noqa: F841 -- the one real holder
    got = ss.describe_holders([obj], depth=3)
    assert any(x.startswith("_Holder.attr <- frame test_the_walk") for x in got), got
    assert not any(x.startswith("list") or "dict[" in x for x in got), got


def test_a_dead_weak_proxy_in_the_heap_does_not_stop_the_scan():
    target = _Payload()
    proxy = weakref.proxy(target)
    del target
    gc.collect()
    keep = [proxy]  # a dead proxy stays reachable, as in PP2's heap
    ss.live_cuda_tensors()  # must not raise ReferenceError
    assert keep


_LOOPS = ("_event_loop_pp_body", "event_loop_pp_disagg_prefill", "event_loop_pp_disagg_decode")


def _loop_source(name):
    from flliper.srt.managers import scheduler_pp_mixin as M

    fn = getattr(M.SchedulerPPMixin, name)
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


@pytest.mark.parametrize("name", _LOOPS)
def test_every_pp_loop_drops_the_forward_result_before_the_next_pass(name):
    """The pass rebinds `pp_outputs` last; `result` must be None by then, so a
    sleep served at the top of the next pass holds no stage output."""
    tree = _loop_source(name)
    body_stmts = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign,))]
    set_pp = [n for n in body_stmts if any(
        isinstance(t, ast.Attribute) and t.attr == "pp_outputs" for t in n.targets)]
    assert set_pp, "no self.pp_outputs rebind found"
    drops = [n for n in body_stmts if any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
             and isinstance(n.value, ast.Constant) and n.value.value is None]
    assert drops, "%s never drops `result`" % name
    last_read = max(n.lineno for n in ast.walk(tree)
                    if isinstance(n, ast.Name) and n.id == "result" and isinstance(n.ctx, ast.Load))
    assert any(last_read < d.lineno <= min(s.lineno for s in set_pp) for d in drops), (
        "`result = None` must sit after the last read (line %d) and before `self.pp_outputs =`" % last_read)
