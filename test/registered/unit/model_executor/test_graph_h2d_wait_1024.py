"""GRAPH-H2D-WAIT-1024 (#237 Teil A, review done/review-pwake-237-1024.md): a prefill CUDA-graph replay joins the
batch's HiCache load-back on the device before the captured body runs.

The only join between a load-back (own non-blocking load_stream, arena gather kernel) and the forward that reads it was
``LayerDoneCounter.wait_until`` from the pools' per-layer accessors -- Python hooks a graph replay never runs (captured
with consumer_index -1). 24.10. logs: int8 27/30/39/18, duo 12/46/16, dual 3 first-forward graph replays per boot that
consumed a load without any layer wait (``PDFLIP-FIRST-FWD-TIMING consumer>=0 layers=0``). Pinned: with a consumer the
runner waits for the producer's last layer (finish_event) BEFORE any replay; without one it waits for nothing.
"""
import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.model_executor.runner import prefill_cuda_graph_runner as R  # noqa: E402


class _Counter:
    def __init__(self, consumer, num_layers=64):
        self.consumer_index, self.num_layers, self.waits = consumer, num_layers, []

    def wait_until(self, threshold):
        self.waits.append(threshold)


class _Pool:
    def __init__(self, counter):
        self.layer_transfer_counter = counter


class _MR:
    def __init__(self, kv=None, req=None):
        self.token_to_kv_pool = kv
        self.req_to_token_pool = req


def test_a_consumed_load_is_joined_at_its_last_layer():
    c = _Counter(1, num_layers=64)
    assert R.join_pending_hicache_load(_MR(kv=_Pool(c))) is True
    assert c.waits == [63], "the whole load (finish_event = last layer event), once"


def test_no_consumer_no_wait():
    c = _Counter(-1)
    assert R.join_pending_hicache_load(_MR(kv=_Pool(c))) is False
    assert c.waits == []


def test_no_counter_no_wait():
    assert R.join_pending_hicache_load(_MR(kv=_Pool(None))) is False
    assert R.join_pending_hicache_load(_MR()) is False


def test_counter_on_the_req_pool_is_found():
    c = _Counter(0, num_layers=48)
    assert R.join_pending_hicache_load(_MR(kv=object(), req=_Pool(c))) is True
    assert c.waits == [47]


class _Stop(Exception):
    pass


def test_execute_joins_before_any_replay():
    """The join runs before the replay session opens (load_batch / backend.replay never see an unjoined load)."""
    order = []
    c = _Counter(2, num_layers=16)
    c.wait_until = lambda t: order.append(("wait", t))

    class _Backend:
        def replay_session(self):
            order.append(("replay_session",))
            raise _Stop()

    fake = type("F", (), {})()
    fake.model_runner = _MR(kv=_Pool(c))
    fake.backend = _Backend()
    with pytest.raises(_Stop):
        R.PrefillCudaGraphRunner.execute(fake, forward_batch=None)
    assert order == [("wait", 15), ("replay_session",)], order


def test_every_backend_replay_in_the_runner_sits_behind_the_join():
    """Structural: each backend.replay call of this runner is inside execute, after the join."""
    src = inspect.getsource(R.PrefillCudaGraphRunner)
    ex = inspect.getsource(R.PrefillCudaGraphRunner.execute)
    assert src.count("self.backend.replay(") == ex.count("self.backend.replay(") >= 1
    j = ex.index("join_pending_hicache_load(self.model_runner)")
    assert j < ex.index("replay_session()") and j < ex.index("self.backend.replay(")
