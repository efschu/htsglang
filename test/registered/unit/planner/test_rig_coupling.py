# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for planner/rig_coupling.py -- the #214 coupling judgement.

Hermetic by construction, and that is also the first thing tested: the module
takes facts and returns a report, so every test here runs with the network
primitives replaced by something that raises. A coupling recommendation that
quietly opened a socket would be a different feature than the one specified,
and the container this runs in cannot reach the fast line anyway.

What is asserted:

  * the gate refuses on the grounds the project MEASURED (the rejected
    register, the runbook's hardware facts) and every row names which;
  * a row that has no basis says ``absent`` and carries the action, rather
    than defaulting to a verdict;
  * an intra-rig or loopback measurement never counts as a wire measurement,
    which is the one dishonesty that would make the transport table useless;
  * a coupling produces a card POOL with lane candidates, never one implied
    verbund;
  * generated commands carry placeholders, not addresses.
"""

import re
import socket
import unittest
import urllib.request
from unittest import mock

from sglang.srt.planner import rejected as rejectedmod
from sglang.srt.planner import rig_coupling as rc

IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _profile(cards, **kw):
    prof = {"cards": cards}
    prof.update(kw)
    return prof


def _rig(rig_id, cards, **kw):
    return rc.RigFacts.from_rig_profile(_profile(cards, **kw), rig_id)


LOCAL_CARDS = [
    {"name": "NVIDIA GeForce RTX 5090", "vram_mib": 32607, "index": 0},
    {"name": "NVIDIA GeForce RTX 3080", "vram_mib": 20480, "index": 1},
]
FAR_CARDS = [{"name": "NVIDIA GeForce RTX 2080 Ti", "vram_mib": 11264, "index": 0}]

VERSIONS = {"nccl": "2.28.9", "torch": "2.11.0+cu130", "ucx": "1.18.0",
            "commit": "abcdef123456"}


class NoNetwork(unittest.TestCase):
    """Every test in this file runs with the network taken away."""

    def setUp(self):
        def _boom(*a, **k):
            raise AssertionError("this module must not touch the network")

        self._patches = [
            mock.patch.object(urllib.request, "urlopen", _boom),
            mock.patch.object(socket, "create_connection", _boom),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])


class TestFacts(NoNetwork):
    def test_arch_inferred_from_card_name_and_labelled_as_inference(self):
        far = _rig("rig-b", FAR_CARDS)
        card = far.cards[0]
        self.assertEqual(card.arch, "sm75")
        self.assertEqual(card.arch_source, "inferred")
        self.assertEqual(card.vendor, "nvidia")

    def test_declared_arch_wins_over_the_name_table(self):
        far = _rig("rig-b", [{"name": "Some OEM card", "arch": "sm89",
                              "vram_mib": 16384}])
        self.assertEqual(far.cards[0].arch, "sm89")
        self.assertEqual(far.cards[0].arch_source, "declared")

    def test_unknown_card_keeps_no_arch_rather_than_guessing(self):
        far = _rig("rig-b", [{"name": "Mystery 9000", "vram_mib": 8192}])
        self.assertIsNone(far.cards[0].arch)
        self.assertEqual(far.cards[0].arch_source, "unknown")

    def test_artifact_import_carries_capabilities_and_fingerprint(self):
        digest = {
            "rig": _profile(FAR_CARDS, **VERSIONS),
            "capabilities": [{"name": "comm/cross_rig", "value": "absent",
                              "provenance": "absent", "note": "no route"}],
            "fingerprint": {"id": "rigfp-deadbeef"},
        }
        facts = rc.RigFacts.from_artifact(digest, "rig-b")
        self.assertEqual(facts.source, "artifact")
        self.assertEqual(facts.nccl, "2.28.9")
        self.assertEqual(facts.capabilities["comm/cross_rig"]["provenance"],
                         "absent")
        self.assertIn("rigfp-deadbeef", " ".join(facts.notes))

    def test_pairing_reach_detail_is_read_without_any_io(self):
        detail = {
            "url": "http://far:8770",
            "identity": {"commit": "abcdef123456", "torch": "2.11.0"},
            "hw_profile": _profile(FAR_CARDS),
            "capabilities": {"capabilities": [
                {"key": "rdma", "state": "available", "reason": "verbs present"}
            ]},
        }
        facts = rc.RigFacts.from_pairing_reach(detail)
        self.assertEqual(facts.source, "pairing-reach")
        self.assertEqual(facts.commit, "abcdef123456")
        self.assertEqual(facts.capabilities["rdma"]["value"], "available")


class TestGate(NoNetwork):
    def _gate(self, **kw):
        local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        far = _rig("rig-b", FAR_CARDS, **VERSIONS)
        return {r.key: r for r in rc.gate(local, far, **kw)}

    def test_verdict_vocabulary_matches_rigmon_compat(self):
        from sglang.srt.rigmon import compat

        self.assertEqual((rc.OK, rc.WARN, rc.BLOCK),
                         (compat.OK, compat.WARN, compat.BLOCK))

    def test_same_tree_passes_and_a_different_tree_blocks(self):
        rows = self._gate()
        self.assertEqual(rows["tree_commit"].verdict, rc.OK)
        local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        far = _rig("rig-b", FAR_CARDS, **dict(VERSIONS, commit="999999999999"))
        row = {r.key: r for r in rc.gate(local, far)}["tree_commit"]
        self.assertEqual(row.verdict, rc.BLOCK)
        self.assertIn("msgspec", row.reason)

    def test_missing_commit_is_absent_not_a_verdict(self):
        local = _rig("rig-a", LOCAL_CARDS, nccl="2.28.9")
        far = _rig("rig-b", FAR_CARDS, nccl="2.28.9")
        row = {r.key: r for r in rc.gate(local, far)}["tree_commit"]
        self.assertEqual(row.provenance, rc.ABSENT)

    def test_nccl_below_the_threshold_only_blocks_a_colocated_plan(self):
        # 2.28.9 is what the venvs pin; the threshold is 2.30.
        self.assertEqual(self._gate()["nccl_colocation"].verdict, rc.WARN)
        blocked = self._gate(colocation_wanted=True)["nccl_colocation"]
        self.assertEqual(blocked.verdict, rc.BLOCK)
        self.assertIn("2.30", blocked.label)
        self.assertTrue(blocked.remedy)

    def test_nccl_above_the_threshold_passes(self):
        local = _rig("rig-a", LOCAL_CARDS, **dict(VERSIONS, nccl="2.30.7"))
        far = _rig("rig-b", FAR_CARDS, **dict(VERSIONS, nccl="2.30.7"))
        row = {r.key: r for r in rc.gate(local, far, colocation_wanted=True)}
        self.assertEqual(row["nccl_colocation"].verdict, rc.OK)

    def test_unreported_nccl_is_absent_rather_than_too_old(self):
        local = _rig("rig-a", LOCAL_CARDS, commit="abcdef123456")
        far = _rig("rig-b", FAR_CARDS, commit="abcdef123456")
        row = {r.key: r for r in rc.gate(local, far)}["nccl_colocation"]
        self.assertEqual(row.provenance, rc.ABSENT)
        self.assertNotEqual(row.verdict, rc.BLOCK)

    def test_gguf_on_a_turing_rank_blocks_from_the_register(self):
        row = self._gate(checkpoint_format="gguf")["arch_gguf_sm75"]
        self.assertEqual(row.verdict, rc.BLOCK)
        self.assertEqual(row.register_key, "gguf_on_sm75")
        entry = rejectedmod.by_key("gguf_on_sm75")
        self.assertEqual(entry.level, rejectedmod.BLOCKED)
        self.assertIn("2080", row.reason)

    def test_without_a_gguf_checkpoint_the_same_wall_is_ok_not_silent(self):
        row = self._gate()["arch_gguf_sm75"]
        self.assertEqual(row.verdict, rc.OK)
        self.assertEqual(row.register_key, "gguf_on_sm75")

    def test_turing_brings_the_dtype_and_backend_rows_with_remedies(self):
        rows = self._gate()
        self.assertEqual(rows["arch_bf16_turing"].verdict, rc.WARN)
        self.assertIn("float16", rows["arch_bf16_turing"].remedy)
        self.assertIn("triton", rows["arch_flashinfer_turing"].remedy)

    def test_arch_walls_are_absent_when_no_turing_card_is_in_the_pool(self):
        local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        far = _rig("rig-b", [{"name": "NVIDIA RTX 4090", "vram_mib": 24564}],
                   **VERSIONS)
        keys = {r.key for r in rc.gate(local, far, checkpoint_format="gguf")}
        self.assertNotIn("arch_bf16_turing", keys)
        self.assertNotIn("arch_gguf_sm75", keys)

    def test_an_inferred_arch_that_fires_says_it_was_inferred(self):
        row = self._gate(checkpoint_format="gguf")["arch_gguf_sm75"]
        self.assertEqual(row.provenance, rc.ESTIMATE)

    def test_an_amd_card_warns_that_nccl_cannot_span_vendors(self):
        local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        far = _rig("rig-b", [{"name": "Radeon RX Vega 64", "vram_mib": 8192}],
                   **VERSIONS)
        row = {r.key: r for r in rc.gate(local, far)}["vendor_mixed"]
        self.assertEqual(row.verdict, rc.WARN)
        self.assertIn("SGLANG_HTCCL", row.remedy)

    def test_transport_row_names_the_broken_verbs_path(self):
        row = self._gate()["transport_available"]
        self.assertEqual(row.verdict, rc.OK)
        self.assertIn("verbs", row.reason)

    def test_transport_row_is_absent_when_neither_rig_reports_ucx(self):
        local = _rig("rig-a", LOCAL_CARDS, nccl="2.30.7")
        far = _rig("rig-b", FAR_CARDS, nccl="2.30.7")
        row = {r.key: r for r in rc.gate(local, far)}["transport_available"]
        self.assertEqual(row.provenance, rc.ABSENT)

    def test_model_fit_is_absent_without_a_checkpoint_and_blocks_with_one(self):
        row = self._gate()["model_fit"]
        self.assertEqual(row.provenance, rc.ABSENT)
        self.assertNotEqual(row.verdict, rc.BLOCK)
        blocked = self._gate(model_fit={"state": "block",
                                        "reason": "does not fit"})["model_fit"]
        self.assertEqual(blocked.verdict, rc.BLOCK)
        self.assertTrue(blocked.remedy)

    def test_the_register_row_for_cross_rig_push_carries_its_evidence(self):
        row = self._gate()["register:crossrig_tp_push"]
        self.assertEqual(row.verdict, rc.WARN)  # not-default, not blocked
        self.assertIn("#244", row.evidence)
        self.assertIn("rig only", row.evidence)

    def test_dmabuf_rdma_row_is_absent_when_nothing_was_probed(self):
        row = self._gate()["dmabuf_rdma"]
        self.assertEqual(row.verdict, rc.WARN)
        self.assertEqual(row.provenance, rc.ABSENT)
        self.assertTrue(row.remedy)

    def test_dmabuf_rdma_row_is_ok_when_every_precondition_is_met(self):
        local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        far = _rig("rig-b", FAR_CARDS, **VERSIONS)
        for facts in (local, far):
            for key, _label in rc.DMABUF_RDMA_CHECKS:
                facts.capabilities[key] = {"value": "ok",
                                           "provenance": "measured", "note": ""}
        row = {r.key: r for r in rc.gate(local, far)}["dmabuf_rdma"]
        self.assertEqual(row.verdict, rc.OK)
        self.assertEqual(row.provenance, rc.MEASURED)
        self.assertIn("not a build", row.reason)

    def test_dmabuf_rdma_row_warns_and_names_the_gap_when_one_side_fails(self):
        local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        far = _rig("rig-b", FAR_CARDS, **VERSIONS)
        for facts in (local, far):
            for key, _label in rc.DMABUF_RDMA_CHECKS:
                facts.capabilities[key] = {"value": "ok"}
        # The one gap the evaluation itself names: KvVmmArena._prop does not
        # yet request the POSIX_FD handle type.
        far.capabilities["dmabuf_vmm_export"] = {"value": False}
        row = {r.key: r for r in rc.gate(local, far)}["dmabuf_rdma"]
        self.assertEqual(row.verdict, rc.WARN)
        self.assertEqual(row.provenance, rc.MEASURED)
        self.assertIn("VMM export", row.reason)

    def test_dmabuf_rdma_row_is_estimate_when_only_partially_probed(self):
        local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        far = _rig("rig-b", FAR_CARDS, **VERSIONS)
        local.capabilities["dmabuf_open_kernel_module"] = {"value": "ok"}
        row = {r.key: r for r in rc.gate(local, far)}["dmabuf_rdma"]
        self.assertEqual(row.verdict, rc.WARN)
        self.assertEqual(row.provenance, rc.ESTIMATE)

    def test_dmabuf_rdma_row_never_blocks(self):
        local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        far = _rig("rig-b", FAR_CARDS, **VERSIONS)
        for facts in (local, far):
            for key, _label in rc.DMABUF_RDMA_CHECKS:
                facts.capabilities[key] = {"value": False}
        row = {r.key: r for r in rc.gate(local, far)}["dmabuf_rdma"]
        self.assertNotEqual(row.verdict, rc.BLOCK)

    def test_dmabuf_rdma_row_treats_the_2080ti_topology_hypothesis_as_a_warn(self):
        # The far rig carries the 2080 Ti (FAR_CARDS); the row must still
        # never block on it, and must say the topology hypothesis is not
        # what decides this, citing the 5090 counter-datum.
        row = self._gate()["dmabuf_rdma"]
        self.assertNotEqual(row.verdict, rc.BLOCK)
        self.assertIn("5090", row.reason)
        self.assertIn("counter-datum", row.reason)
        self.assertIn("root complex", row.reason)

    def test_dmabuf_rdma_row_is_wired_into_the_gate(self):
        self.assertIn("dmabuf_rdma", self._gate())

    def test_every_row_that_is_not_ok_carries_a_reason(self):
        for row in self._gate(checkpoint_format="gguf",
                              colocation_wanted=True).values():
            if row.verdict != rc.OK:
                self.assertTrue(row.reason, row.key)
                self.assertIn(row.provenance,
                              (rc.MEASURED, rc.ESTIMATE, rc.ABSENT), row.key)


class TestTransportPlan(NoNetwork):
    def setUp(self):
        super().setUp()
        self.local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        self.far = _rig("rig-b", FAR_CARDS, **VERSIONS)

    def _by_class(self, **kw):
        return {t.message_class: t
                for t in rc.transport_plan(self.local, self.far, **kw)}

    def test_all_four_classes_are_reported(self):
        self.assertEqual(set(self._by_class()),
                         {m["key"] for m in rc.MESSAGE_CLASSES})

    def test_nothing_measured_means_absent_plus_the_action(self):
        rows = self._by_class()
        for key in ("tp_small", "tp_bulk"):
            row = rows[key]
            self.assertEqual(row.provenance, rc.ABSENT, key)
            self.assertTrue(row.how_to_measure, key)
        # kv_bulk has exactly one carrier, so it is an estimate rather than
        # absent -- but it still carries how it would be measured.
        self.assertEqual(rows["kv_bulk"].provenance, rc.ESTIMATE)
        self.assertTrue(rows["kv_bulk"].how_to_measure)

    def test_the_control_plane_stays_on_the_slow_lan_by_design(self):
        row = self._by_class()["control"]
        self.assertEqual(row.chosen, "gloo-tcp")
        self.assertEqual(row.provenance, rc.MEASURED)

    def test_a_wire_row_makes_the_small_class_measured_with_its_id(self):
        digest = {"measurements": [{
            "id": "comm/cross_rig/all_reduce/20KiB", "label": "wire",
            "unit": "us", "value": 166.2,
            "context": {"op": "all_reduce", "backend": "htccl-ucx",
                        "size_kib": 20}}]}
        row = self._by_class(remote_digest=digest)["tp_small"]
        self.assertEqual(row.provenance, rc.MEASURED)
        self.assertEqual(row.chosen, "htccl-ucx")
        self.assertEqual(row.evidence[0]["id"], "comm/cross_rig/all_reduce/20KiB")
        self.assertFalse(row.how_to_measure)

    def test_a_loopback_row_is_not_a_wire_row(self):
        # The whole point: the comm suite's ucx arm runs over loopback in the
        # container. Counting it as a link measurement would put a number on
        # the wire that never crossed it.
        digest = {"measurements": [{
            "id": "comm/collective_htccl_ucx/all_reduce/20KiB",
            "unit": "us", "value": 26.6,
            "context": {"op": "all_reduce", "backend": "htccl_ucx"}}]}
        row = self._by_class(local_digest=digest)["tp_small"]
        self.assertEqual(row.provenance, rc.ABSENT)
        self.assertIsNone(row.chosen)

    def test_a_row_tagged_cross_rig_counts_even_without_the_arm_prefix(self):
        digest = {"measurements": [{
            "id": "link/all_reduce/80KiB", "unit": "GB/s", "value": 2.07,
            "context": {"pair": "cross-rig", "backend": "nccl-sockets"}}]}
        row = self._by_class(remote_digest=digest)["tp_bulk"]
        self.assertEqual(row.provenance, rc.MEASURED)
        self.assertEqual(row.chosen, "nccl-sockets")

    def test_the_bulk_class_says_it_shares_a_context_with_the_small_one(self):
        self.assertIn("Not separable", self._by_class()["tp_bulk"].note)

    def test_verbs_is_listed_as_unavailable_with_the_failure_named(self):
        opts = {c["key"]: c for c in self._by_class()["tp_small"].candidates}
        self.assertEqual(opts["nccl-verbs"]["verdict"], "unavailable")
        self.assertIn("IBV_WC_REM_INV_REQ_ERR", opts["nccl-verbs"]["needs"])

    def test_ucx_missing_on_one_side_is_unknown_not_usable(self):
        far = _rig("rig-b", FAR_CARDS, nccl="2.28.9", commit="abcdef123456")
        rows = {t.message_class: t for t in rc.transport_plan(self.local, far)}
        opts = {c["key"]: c for c in rows["tp_small"].candidates}
        self.assertEqual(opts["htccl-ucx"]["verdict"], "unknown")

    def test_a_single_available_carrier_is_an_estimate_not_a_measurement(self):
        rows = self._by_class()
        kv = rows["kv_bulk"]
        self.assertEqual(kv.chosen, "mooncake")
        self.assertEqual(kv.provenance, rc.ESTIMATE)
        self.assertIn("trivial", kv.reason)

    def test_the_intra_rig_pair_matrix_is_context_never_a_wire_verdict(self):
        rows = self._by_class(pair_matrix=[{"src_uuid": "a", "dst_uuid": "b",
                                            "bandwidth_gbs": 4.32}])
        row = rows["tp_small"]
        ctx = [c for c in row.candidates if c["key"] == "intra-rig-pairs"]
        self.assertEqual(len(ctx), 1)
        self.assertEqual(ctx[0]["verdict"], "context")
        self.assertEqual(row.provenance, rc.ABSENT)


class TestPoolAndLanes(NoNetwork):
    def setUp(self):
        super().setUp()
        self.local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        self.far = _rig("rig-b", FAR_CARDS, **VERSIONS)

    def test_the_pool_lists_every_card_of_both_rigs(self):
        pool = rc.card_pool(self.local, self.far)
        self.assertEqual(pool["card_count"], 3)
        self.assertEqual({c["rig"] for c in pool["cards"]}, {"rig-a", "rig-b"})

    def test_lanes_are_candidates_and_a_cross_lane_is_only_one_of_them(self):
        pool = rc.card_pool(self.local, self.far)
        lanes = {ln["key"]: ln for ln in pool["lane_candidates"]}
        self.assertEqual(set(lanes), {"lane-rig-a", "lane-rig-b", "lane-cross"})
        self.assertEqual(len(lanes["lane-cross"]["cards"]), 3)
        self.assertTrue(all(ln["exclusive"] for ln in lanes.values()))

    def test_a_block_hits_the_cross_lane_only(self):
        rows = rc.gate(self.local, self.far, checkpoint_format="gguf")
        pool = rc.card_pool(self.local, self.far, rows)
        lanes = {ln["key"]: ln for ln in pool["lane_candidates"]}
        self.assertIn("arch_gguf_sm75", lanes["lane-cross"]["blocked_by"])
        self.assertFalse(lanes["lane-rig-a"]["blocked_by"])

    def test_a_rig_with_no_cards_produces_no_lane_for_itself(self):
        empty = rc.RigFacts(rig_id="rig-b")
        pool = rc.card_pool(self.local, empty)
        keys = {ln["key"] for ln in pool["lane_candidates"]}
        self.assertEqual(keys, {"lane-rig-a"})


class TestHostSteps(NoNetwork):
    def test_no_command_contains_a_real_address(self):
        # 0.0.0.0 and 127.0.0.1 name no machine; anything else literal would.
        for step in rc.host_steps("far-rig:8770"):
            for line in step.command.splitlines():
                for hit in IPV4.findall(line):
                    self.assertIn(hit, ("0.0.0.0", "127.0.0.1"), line)

    def test_every_host_step_says_where_it_runs_and_why_not_here(self):
        for step in rc.host_steps():
            self.assertIn(step.where, (rc.HERE, rc.HOST, rc.FAR))
            self.assertTrue(step.why_not_here, step.key)
            self.assertTrue(step.command, step.key)

    def test_the_link_measurement_is_host_only_and_says_so(self):
        step = {s.key: s for s in rc.host_steps()}["link_cost"]
        self.assertEqual(step.where, rc.HOST)
        self.assertIn("/dev/infiniband", step.why_not_here)
        self.assertIn("link_collective_cost.py", step.command)

    def test_commands_use_the_rig_env_placeholder_convention(self):
        blob = "\n".join(s.command for s in rc.host_steps())
        self.assertIn("source /root/rig-env.sh", blob)
        self.assertIn("${RIG2_HOST:-<rig2-host>}", blob)


class TestCouple(NoNetwork):
    def _report(self, **kw):
        local = _rig("rig-a", LOCAL_CARDS, **VERSIONS)
        far = _rig("rig-b", FAR_CARDS, **VERSIONS)
        return rc.couple(local, far, target="far-rig:8770", **kw)

    def test_the_payload_states_that_it_contacted_and_booted_nothing(self):
        out = self._report().to_json()
        self.assertTrue(out["contacts_nothing"])
        self.assertTrue(out["boots_nothing"])

    def test_a_blocking_row_makes_the_whole_verdict_block(self):
        out = self._report(checkpoint_format="gguf").to_json()
        self.assertEqual(out["verdict"], rc.BLOCK)
        self.assertIn("intra-rig lanes are unaffected", out["summary"])

    def test_without_a_block_the_verdict_carries_the_conditions(self):
        out = self._report().to_json()
        self.assertEqual(out["verdict"], rc.WARN)
        self.assertIn("message classes have a measured basis", out["summary"])

    def test_the_cuda_graph_row_follows_the_chosen_transport(self):
        digest = {"measurements": [{
            "id": "comm/cross_rig/all_reduce/20KiB", "unit": "us",
            "value": 166.2, "context": {"backend": "htccl-ucx"}}]}
        rows = {r["key"]: r for r in
                self._report(remote_digest=digest).to_json()["gate"]}
        self.assertEqual(rows["cuda_graph"]["verdict"], rc.WARN)
        self.assertIn("--disable-cuda-graph", rows["cuda_graph"]["remedy"])

    def test_the_control_plane_alone_does_not_forbid_cuda_graphs(self):
        # gloo carries rendezvous, not collectives inside a capture. A
        # warning that fires on every plan would say nothing.
        rows = {r["key"]: r for r in self._report().to_json()["gate"]}
        self.assertEqual(rows["cuda_graph"]["verdict"], rc.OK)
        self.assertEqual(rows["cuda_graph"]["provenance"], rc.ABSENT)

    def test_the_report_is_json_serialisable(self):
        import json

        json.dumps(self._report().to_json())


class TestWebuiAdapter(NoNetwork):
    def setUp(self):
        super().setUp()
        from sglang.srt.planner import webui

        self.webui = webui
        # No cached probe, so the adapter's optional inputs are exercised in
        # their absent form rather than against this machine's cache.
        p = mock.patch("sglang.srt.rigmon.card_probe.load_card_probe",
                       return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def _artifacts(self):
        return ({"rig": _profile(LOCAL_CARDS, **VERSIONS)},
                {"rig": _profile(FAR_CARDS, **VERSIONS)})

    def test_a_plan_from_two_artifacts_needs_no_session_and_no_io(self):
        local, remote = self._artifacts()
        out = self.webui.rig_coupling_plan_payload(
            {"local_artifact": local, "remote_artifact": remote})
        self.assertTrue(out["ok"])
        self.assertTrue(out["have_remote"])
        self.assertEqual(out["pool"]["card_count"], 3)
        self.assertTrue(out["gate"])

    def test_without_a_far_rig_the_answer_is_the_host_steps(self):
        local, _ = self._artifacts()
        out = self.webui.rig_coupling_plan_payload({"local_artifact": local})
        self.assertTrue(out["ok"])
        self.assertFalse(out["have_remote"])
        self.assertTrue(out["host_steps"])
        self.assertNotIn("gate", out)

    def test_an_unknown_session_is_an_error_not_a_silent_empty_plan(self):
        local, _ = self._artifacts()
        out = self.webui.rig_coupling_plan_payload(
            {"local_artifact": local, "session_id": "nosuchsession"})
        self.assertFalse(out["ok"])
        self.assertIn("no such pairing session", out["error"])

    def test_a_session_that_has_not_reached_yet_is_refused_with_the_remedy(self):
        from sglang.srt.rigmon import pairing

        local, _ = self._artifacts()
        store = pairing.PairingStore()
        session = store.create("far-rig:8770")
        with mock.patch.object(pairing, "STORE", store):
            out = self.webui.rig_coupling_plan_payload(
                {"local_artifact": local, "session_id": session.session_id})
        self.assertFalse(out["ok"])
        self.assertIn("never contacts", out["remedy"])

    def test_a_reached_session_feeds_the_plan_without_re_fetching(self):
        from sglang.srt.rigmon import pairing

        local, _ = self._artifacts()
        store = pairing.PairingStore()
        session = store.create("far-rig:8770")
        session.steps["reach"] = pairing.StepResult(
            step="reach", state=pairing.OK,
            detail={"url": "http://far-rig:8770", "reachable": True,
                    "identity": {"commit": VERSIONS["commit"]},
                    "hw_profile": _profile(FAR_CARDS, **VERSIONS)})
        with mock.patch.object(pairing, "STORE", store):
            out = self.webui.rig_coupling_plan_payload(
                {"local_artifact": local, "session_id": session.session_id})
        self.assertTrue(out["ok"])
        self.assertEqual(out["target"], "far-rig:8770")
        self.assertEqual(out["remote"]["source"], "pairing-reach")

    def test_the_route_is_registered_on_the_handler(self):
        import inspect

        src = inspect.getsource(self.webui._Handler.do_POST)
        self.assertIn("/api/rig_coupling/plan", src)

    def test_the_page_renders_the_plan_and_derives_nothing(self):
        html = self.webui.INDEX_HTML
        self.assertIn("coupling plan", html)
        self.assertIn("couplingPlan()", html)
        self.assertIn("/api/rig_coupling/plan", html)
        # The verdict words come from the server; the page must not carry a
        # second copy of the gate's reasoning.
        self.assertNotIn("IBV_WC_REM_INV_REQ_ERR", html)


if __name__ == "__main__":
    unittest.main()
