"""Review S22 (18.10.) probe -- two channel properties the S22 tests do not hold (both mutants survived 14/14).

* MA: the dead-process filter of ``_read_worker_sets``. The S22 test writes the dead file in the PART-1 format (a bare
  list); since part 2 the reader wants ``{"ev", "nd"}``, so that file fails to parse and is skipped whether the filter
  is there or not. Here: the dead file in the CURRENT format, and a second LIVE publisher (union).
* MB: the ``children`` clause of the ``nd`` set (a worker cannot delete an INTERIOR node). The S22 nd test builds two
  sibling leaves only. Metal h16v4b 02:09:44/02:18:01: five ``K23 SPILL-DIVERGED outcome=unreconcilable`` were exactly
  this shape (worker: device-evicted AND interior).
"""
import json
import os
import sys
import threading

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_pdflip_read_room_d_1015 import FULL, _clean, _real_d_tree, evict_mock, roles  # noqa: E402,F401
from test_s22_r12_spill_class_1018 import chan  # noqa: E402,F401

from flliper.srt.mem_cache import form_a_host_shadow as R12  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import InsertParams  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402


def test_probe_a_dead_publisher_in_the_current_format_is_ignored_and_live_ones_unite(roles, evict_mock, chan):
    me = threading.current_thread().name
    worker = _real_d_tree([(1, list(range(8)))], None, "worker")
    roles.now[me] = "worker"
    R12.register_tree(worker)
    worker._t_nodes[1].component_data[FULL].value = None
    R12._dev_refresh(worker)                                   # this process: one live publisher
    roles.now[me] = "host"
    pre = R12._dev_dir_prefix(R12._tree_arena(worker))
    with open(pre + "999999999", "w") as f:                    # a worker process that no longer exists
        json.dump({"ev": [["zz", 1]], "nd": [["yy", 2]]}, f)
    with open(pre + str(os.getppid()), "w") as f:              # a second LIVE process (the union)
        json.dump({"ev": [["ww", 3]], "nd": []}, f)
    ev, nd = R12.worker_evicted_keys(worker), R12.worker_nodelete_keys(worker)
    assert R12.node_key(worker._t_nodes[1]) in ev and ("ww", 3) in ev, ev
    assert ("zz", 1) not in ev and ("yy", 2) not in nd, (ev, nd)


def test_probe_nodelete_holds_an_interior_device_evicted_worker_node(roles, evict_mock, chan):
    me = threading.current_thread().name
    worker = _real_d_tree([(1, list(range(8)))], None, "worker")
    parent = worker._t_nodes[1]
    worker.insert(InsertParams(key=RadixKey(list(range(1, 13)), None),
                               value=worker._t_alloc.alloc(12).to(dtype=torch.int64)))
    kids = list(parent.children.values())
    assert len(kids) == 1, "fixture: the longer key must hang below the first node"
    child = kids[0]
    child.hash_value = [f"c{i}" for i in range(len(child.key))]
    for n in (parent, child):
        n.component_data[FULL].value = None                    # the worker peeled both: parent interior, child leaf
    roles.now[me] = "worker"
    R12.register_tree(worker)
    R12._dev_refresh(worker)
    roles.now[me] = "host"
    nd, ev = R12.worker_nodelete_keys(worker), R12.worker_evicted_keys(worker)
    assert {R12.node_key(parent), R12.node_key(child)} <= ev, ev
    assert R12.node_key(parent) in nd, "an interior worker node is not deletable there (K23 unreconcilable)"
    assert R12.node_key(child) not in nd, nd
