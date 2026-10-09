# SPDX-License-Identifier: Apache-2.0
"""PRIORITY LANES, part L5 (integration) -- plan deskq/PLAN-PRIO-LANES-1008.md section 3, row L5.

L2 (D side) and L3 (P side) were built on separate branches and each defined its own half of ``POST /weg2/lane_floor``:
a request class ``Weg2LaneFloorReqInput``, a route, a dispatcher row and a scheduler handler ``handle_weg2_lane_floor``.
Merged side by side, the second definition of each wins in Python and the first half silently never runs (the
dispatcher builds one handler per type; two ``def`` of one name in a class keep the last). The merge keeps ONE of each:

* ONE request class in ``io_struct`` (a second ``class`` of the same name would shadow the first);
* ONE route in ``http_server`` (the answer is L2's: floor / epoch / requeued / held / message);
* ONE dispatcher row and ONE ``Scheduler.handle_weg2_lane_floor`` that runs BOTH halves: ``lanes_p.on_rpc`` (group P: PP0
  records the value, stamps it on the PP-room vote) and ``d_park_runtime.lane_floor`` (group D: admission floor, requeue of
  the held parks) and returns the L2 output on either group.

CAN-FAIL: each source-shape check names the duplicate it forbids (the merge result before L5 had every one of them), and
the behaviour tests drive the REAL handler once as group P and once as group D.
"""
from __future__ import annotations

import ast
import os
import types
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import io_struct as io  # noqa: E402
from sglang.srt.managers import scheduler as sch  # noqa: E402
from sglang.srt.weg2 import d_lane, lanes, lanes_p  # noqa: E402

_SRT = os.path.join(os.path.dirname(io.__file__), "..")


def _src(*parts):
    with open(os.path.join(_SRT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _class_names(tree, container=None):
    node = tree if container is None else container
    return [n.name for n in node.body if isinstance(n, ast.ClassDef)]


def _func_names(cls):
    return [n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


class _Env:
    """SGLANG_WEG2_LANES=1 and one group letter (what a lane boot has), restored on exit."""

    KEYS = ("SGLANG_WEG2_LANES", "SGLANG_WEG2_GROUP", "SGLANG_WEG2_PARK")

    def __init__(self, group, park=None):
        self.group, self.park = group, park

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.KEYS}
        os.environ["SGLANG_WEG2_LANES"] = "1"
        os.environ["SGLANG_WEG2_GROUP"] = self.group
        if self.park is None:
            os.environ.pop("SGLANG_WEG2_PARK", None)
        else:
            os.environ["SGLANG_WEG2_PARK"] = self.park
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class OneOfEachTest(unittest.TestCase):
    def test_one_request_class_in_io_struct(self):
        names = _class_names(ast.parse(_src("managers", "io_struct.py")))
        self.assertEqual(names.count("Weg2LaneFloorReqInput"), 1)
        self.assertEqual(names.count("Weg2LaneFloorReqOutput"), 1)

    def test_one_route_one_function_one_import_in_http_server(self):
        src = _src("entrypoints", "http_server.py")
        self.assertEqual(src.count('@app.api_route("/weg2/lane_floor", methods=["POST"])'), 1)
        tree = ast.parse(src)
        fns = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertEqual(fns.count("weg2_lane_floor"), 1)
        for node in tree.body:  # the `from sglang.srt.managers.io_struct import (...)` list names it once
            if isinstance(node, ast.ImportFrom) and node.module == "sglang.srt.managers.io_struct":
                got = [a.name for a in node.names]
                self.assertLessEqual(got.count("Weg2LaneFloorReqInput"), 1)

    def test_one_dispatcher_row_one_handler_in_scheduler(self):
        src = _src("managers", "scheduler.py")
        self.assertEqual(src.count("(Weg2LaneFloorReqInput, self.handle_weg2_lane_floor)"), 1)
        tree = ast.parse(src)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        self.assertEqual(_func_names(cls).count("handle_weg2_lane_floor"), 1)

    def test_the_handler_runs_both_halves_and_returns_the_d_answer(self):
        src = _src("managers", "scheduler.py")
        at = src.index("    def handle_weg2_lane_floor(self, recv_req):")
        body = src[at:src.index("    def _weg2_lane_skip", at)]
        self.assertIn("lanes_p.on_rpc(self, recv_req)", body)
        self.assertIn("return d_park_runtime.lane_floor(self, recv_req)", body)
        self.assertLess(body.index("        lanes_p.on_rpc(self, recv_req)"),
                        body.index("        return d_park_runtime.lane_floor(self, recv_req)"))  # the code lines, not the docstring


def _stage(rank, pp_size=3):
    return SimpleNamespace(ps=SimpleNamespace(pp_size=pp_size, pp_rank=rank), waiting_queue=[])


class HandlerBehaviourTest(unittest.TestCase):
    def test_group_p_pp0_records_the_floor_and_answers(self):
        with _Env("P", park="1"):
            s = _stage(0)
            out = sch.Scheduler.handle_weg2_lane_floor(s, io.Weg2LaneFloorReqInput(floor=2, epoch=5))
            self.assertIsNotNone(out)
            self.assertTrue(out.success)  # not group D: success, nothing applied on the D side
            stamp = lanes_p.pp0_pass(s)
            self.assertEqual((stamp.floor, stamp.epoch), (2, 5))
            self.assertFalse(hasattr(s, d_lane.FLOOR_ATTR) and getattr(s, d_lane.FLOOR_ATTR))  # D's attribute untouched

    def test_group_p_follower_takes_no_floor_from_the_rpc_itself(self):
        with _Env("P", park="1"):
            s = _stage(1)
            out = sch.Scheduler.handle_weg2_lane_floor(s, io.Weg2LaneFloorReqInput(floor=2, epoch=5))
            self.assertTrue(out.success)
            self.assertIsNone(lanes_p.pp0_pass(s))  # PP1 records nothing; it gets the stamp on the chain

    def test_group_d_applies_the_admission_floor_and_p_state_stays_empty(self):
        with _Env("D"):
            s = types.SimpleNamespace(weg2_d_parked=[], waiting_queue=[], ps=SimpleNamespace(pp_size=1, pp_rank=0))
            s._add_request_to_queue = lambda *a, **k: None
            out = sch.Scheduler.handle_weg2_lane_floor(s, io.Weg2LaneFloorReqInput(floor=1, epoch=3))
            self.assertTrue(out.success, out.message)
            self.assertEqual((out.floor, out.epoch), (1, 3))
            self.assertEqual((getattr(s, d_lane.FLOOR_ATTR), getattr(s, d_lane.EPOCH_ATTR)), (1, 3))
            self.assertIsNone(getattr(s, lanes_p.STATE_ATTR, None))  # the P half did nothing on group D

    def test_switch_off_is_a_refusal_naming_the_switch_and_no_state(self):
        old = {k: os.environ.pop(k, None) for k in _Env.KEYS}
        try:
            s = _stage(0)
            out = sch.Scheduler.handle_weg2_lane_floor(s, io.Weg2LaneFloorReqInput(floor=1, epoch=1))
            self.assertFalse(out.success)
            self.assertIn("SGLANG_WEG2_LANES", out.message)
            self.assertIsNone(getattr(s, lanes_p.STATE_ATTR, None))
            self.assertFalse(lanes.enabled())
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v


class BodiesMeetTheEndpointsTest(unittest.TestCase):
    """What L4 SENDS is what L2 / L3 RECEIVE: the route validates the JSON body through pydantic (``BaseReq`` is an
    array-like msgspec struct with a pydantic core schema, so ``msgspec.convert`` of a dict does not apply), extra keys
    are dropped, the named ones must land."""

    def test_l4_park_body_lands_in_the_l2_request(self):
        from pydantic import TypeAdapter

        from sglang.srt.weg2 import phase_policy as pp

        body = pp.lane_park_body(3, ["a", "b"], 2, 5)
        req = TypeAdapter(io.Weg2ParkRunningReqInput).validate_python(body)
        self.assertEqual(list(req.rids), ["a", "b"])
        self.assertEqual(req.hold, "lane")  # without hold + rids an old D would park every running request
        self.assertEqual(req.epoch, 3)

    def test_l4_floor_body_lands_in_the_one_floor_request(self):
        from pydantic import TypeAdapter

        body = {lanes.RPC_KEY_FLOOR: 2, lanes.RPC_KEY_EPOCH: 7}
        req = TypeAdapter(io.Weg2LaneFloorReqInput).validate_python(body)
        self.assertEqual((req.floor, req.epoch), (2, 7))


class LaneEnvsAreNfOnlyInTheCatalogTest(unittest.TestCase):
    """The lane envs exist on the NF line only (the 27B revision 84d04adae1 has none of them).  The shipped catalog
    must say so (entries and edges K135/K136 carry baeume ['nf']), else the dashboard offers a switch on 27B that has
    no effect.  CAN-FAIL: the catalog rebuilt from 27B archive + hand-applied L1 hunk listed ['27b', 'nf']."""

    NAMES = ("SGLANG_WEG2_LANES", "SGLANG_WEG2_LANE_KEEPALIVE_S", "SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS")

    def _catalog(self):
        import json

        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        with open(os.path.join(root, "tools", "rig_dashboard", "rigdash", "profil_data", "catalog.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def test_catalog_entries_name_only_the_nf_tree(self):
        cat = self._catalog()
        for name in self.NAMES:
            self.assertEqual(cat["entries"][name]["baeume"], ["nf"], name)
            self.assertEqual(cat["entries"][name]["source"]["baum"], "nf", name)

    def test_edges_k135_k136_name_only_the_nf_tree(self):
        import json

        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "..", "..", "..", "python", "sglang", "srt", "weg2", "kantenkatalog_1004.json")
        with open(path, encoding="utf-8") as fh:
            kanten = {k["id"]: k for k in json.load(fh)["kanten"]}
        for kid in ("K135", "K136"):
            self.assertEqual(kanten[kid].get("baeume"), ["nf"], kid)
        for name in self.NAMES[1:]:
            deps = self._catalog()["entries"][name]["depends"]
            self.assertTrue(deps and all(d.get("baeume") == ["nf"] for d in deps), name)


if __name__ == "__main__":
    unittest.main()
