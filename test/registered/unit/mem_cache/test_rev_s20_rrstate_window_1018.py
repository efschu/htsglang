"""Review S20 (18.10.) probe: a READ-ROOM STATE that stays pending on a worker re-enters ``_S.pending`` and is
re-applied at EVERY following pass (``_consume_apply`` -> ``apply(tree, [])``). review readroomd2 A4 asked for the
window to be named per OCCURRENCE; ``_rrstate_diverged`` fires on every re-apply instead (unthrottled WARNING, the
counter counts passes, not windows). Expected RED on c75f9c2699: 5 lines / n=5 for one window of 5 passes."""

import logging
import threading

import test_weg2_read_room_d_1015 as DD
from test_weg2_read_room_d_1015 import _clean, roles  # noqa: F401 -- fixtures


def test_one_pending_window_is_named_once_not_per_pass(tmp_path, roles, caplog):  # noqa: F811
    rr = DD._rr()
    me = threading.current_thread().name
    root = tmp_path / "store"
    root.mkdir()
    for s_ in range(8):
        (root / f"s{s_}.bin").write_bytes(b"\x01")
    spans = [(1, list(range(8)))]
    tp0 = DD._real_d_tree(spans, None, "tp0")
    worker = DD._real_d_tree(spans, None, "worker")
    roles.now[me] = "host"
    rr.form_a_spill(tp0, DD._Pool(None, {s_: f"s{s_}" for s_ in range(8)}, str(root)), 1)
    worker._t_nodes[1].component_data[DD.FULL].host_lock_ref = 1   # the worker's own store write holds the rows
    with caplog.at_level(logging.WARNING):
        for _ in range(5):                                          # five passes of the same window
            DD._broadcast(roles, tp0, worker)
    msgs = [r.getMessage() for r in caplog.records if "READ-ROOM-STATE-DIVERGED" in r.getMessage()]
    worker._t_nodes[1].component_data[DD.FULL].host_lock_ref = 0
    DD._broadcast(roles, tp0, worker)
    assert DD._released(worker) == [1]                              # the window closes (as R2_5b)
    assert len(msgs) == 1, f"one pending window named {len(msgs)}x (counter n={DD.R12._S.counts.get('rrstate_diverged')})"
