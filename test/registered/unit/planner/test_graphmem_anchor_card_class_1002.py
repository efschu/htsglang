"""SM89-DURCHSPIEL-1002 / HW-GENERISCH 1002 K8: the graph-mem anchor key
carries the card class/arch.

The anchor store is machine-local, but keys used to name only the CONFIG
(model/tp/spec/kv/attn/page/bs). A foreign rig -- e.g. an sm_89 Ada box --
whose boot logs landed in the same store (or whose config mirrors a measured
one) got the reference rig's measured capture numbers handed over with
provenance "measured". K8: the key gains a ``cards:`` segment (the
``weg2.card_identity`` class labels; ``?`` when NVML cannot answer), so an
anchor is only ever claimed by a boot on the card class it was measured on.

Red on the base tree: base ``anchor_key`` has no ``cards:`` segment (the
version is ``v2``), so the sm_89 lookup below HITS the reference-rig anchor.
"""

import os
import tempfile
import unittest

from sglang.srt.planner import graphmem

RIG = ("RTX5090", "RTX3080", "RTX3080")
ADA = ("RTX4090/24564MiB/sm89", "RTX4090/24564MiB/sm89", "RTX4090/24564MiB/sm89")


def _meta(**over):
    meta = {
        "model_path": "/models/Qwen3.6-27B-FP8",
        "tp_size": 3,
        "kv_cache_dtype": "fp8_e5m2",
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_num_draft_tokens": 4,
        "speculative_adaptive": True,
        "attention_backend": "flashinfer",
        "page_size": 64,
        "decode_bs": [1, 2, 3, 4],
    }
    meta.update(over)
    return meta


_SUMMARY = {
    "per_rank_mib": {"0": 300.0, "1": 300.0, "2": 300.0},
    "items": [{"label": "target decode", "total_mib": 900.0}],
}


class TestAnchorCardClass(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = graphmem.AnchorStore(os.path.join(self.tmp.name, "anchors.json"))

    def test_key_names_the_card_class_and_bumped_version(self):
        key = graphmem.anchor_key(_meta(card_classes=RIG))
        self.assertTrue(graphmem.ANCHOR_KEY_VERSION.startswith("v3"),
                        "anchor key version must change with the key fields")
        # sorted labels: the segment is order-independent on purpose
        self.assertIn("cards:RTX3080+RTX3080+RTX5090", key)

    def test_reference_anchor_does_not_answer_an_sm89_lookup(self):
        self.store.record(_meta(card_classes=RIG), _SUMMARY, source="test")
        # same config on Ada: NOT the same measurement
        self.assertIsNone(self.store.lookup(_meta(card_classes=ADA)))
        # same config on the measured class: still measured
        hit = self.store.lookup(_meta(card_classes=RIG))
        self.assertIsNotNone(hit)
        self.assertEqual(hit["provenance"], "measured")

    def test_unknown_cards_anchor_does_not_answer_a_known_class_lookup(self):
        # a store populated where NVML could not answer (``?``) stays unknown:
        # a rig that states its cards must not inherit it as "measured".
        self.store.record(_meta(), _SUMMARY, source="test")
        self.assertIn("cards:?", graphmem.anchor_key(_meta()))
        self.assertIsNone(self.store.lookup(_meta(card_classes=RIG)))

    def test_live_card_classes_reads_the_replay_seam(self):
        from sglang.srt.registry import nvml as nvml_registry

        rows = [
            {"index": 0, "uuid": "GPU-a", "name": "NVIDIA GeForce RTX 3080",
             "total_bytes": 20480 * (1 << 20), "reserved_bytes": 0,
             "pci_bus_id": "0000:01:00.0", "cc_major": 8, "cc_minor": 6},
            {"index": 1, "uuid": "GPU-b", "name": "NVIDIA GeForce RTX 5090",
             "total_bytes": 32607 * (1 << 20), "reserved_bytes": 0,
             "pci_bus_id": "0000:02:00.0", "cc_major": 12, "cc_minor": 0},
        ]
        import json

        path = os.path.join(self.tmp.name, "nvml.json")
        with open(path, "w") as fh:
            json.dump(rows, fh)
        old = os.environ.get(nvml_registry.ENV_NVML_REPLAY)
        os.environ[nvml_registry.ENV_NVML_REPLAY] = path
        try:
            self.assertEqual(graphmem.live_card_classes(), ("RTX3080", "RTX5090"))
        finally:
            if old is None:
                os.environ.pop(nvml_registry.ENV_NVML_REPLAY, None)
            else:
                os.environ[nvml_registry.ENV_NVML_REPLAY] = old


if __name__ == "__main__":
    unittest.main()
