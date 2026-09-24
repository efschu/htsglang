import pickle
import unittest
from collections import OrderedDict, defaultdict, deque
from functools import partial
from types import SimpleNamespace

import torch

from sglang.srt.utils.common import MultiprocessingSerializer, safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSafeUnpickler(CustomTestCase):
    def test_rejects_dangerous_builtin_globals(self):
        for name in ("__import__", "getattr", "eval", "exec", "compile", "open"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    RuntimeError, rf"Blocked unsafe global \(builtins\.{name}\)"
                ),
            ):
                # GLOBAL resolves the callable but does not invoke it. This exercises
                # the deserialization boundary without constructing an exploit chain.
                safe_pickle_loads(f"cbuiltins\n{name}\n.".encode())

    def test_rejects_unlisted_standard_library_globals(self):
        for module, name in (
            ("copyreg", "_reconstructor"),
            ("operator", "attrgetter"),
            ("types", "FunctionType"),
        ):
            with (
                self.subTest(module=module, name=name),
                self.assertRaisesRegex(
                    RuntimeError, rf"Blocked unsafe global \({module}\.{name}\)"
                ),
            ):
                safe_pickle_loads(f"c{module}\n{name}\n.".encode())

    def test_round_trips_safe_standard_library_types(self):
        value = SimpleNamespace(
            values=OrderedDict([("items", deque([1, 2]))]),
            factory=defaultdict(list, {"items": [3]}),
            index=slice(1, 4),
            parser=partial(int, base=10),
        )

        restored = safe_pickle_loads(
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        )

        self.assertEqual(restored.values, value.values)
        self.assertEqual(restored.factory, value.factory)
        self.assertEqual(restored.index, value.index)
        self.assertEqual(restored.parser("11"), 11)

    def test_round_trips_tensor_payload(self):
        value = [("weight", torch.arange(6).reshape(2, 3))]

        restored = MultiprocessingSerializer.deserialize(
            MultiprocessingSerializer.serialize(value)
        )

        self.assertEqual(restored[0][0], "weight")
        self.assertTrue(torch.equal(restored[0][1], value[0][1]))


class TestSafeUnpicklerForkLayout(CustomTestCase):
    """Fork-specific guards for the explicit-globals allowlist (#39858/#40259).

    The upstream allowlist names module paths of the upstream tree
    (disaggregation/encoder/receiver.py, model_runner_components/...). An
    entry that does not import in THIS tree silently turns a working payload
    into a "Blocked unsafe global" refusal, so every sglang entry must
    resolve here.
    """

    # Defined only inside the NPU branch of patch_torch (same as upstream).
    _NPU_ONLY = {("sglang.srt.utils.patch_torch", "_rebuild_npu_tensor_modified")}

    def test_allowlisted_sglang_globals_resolve_in_this_tree(self):
        import importlib

        from sglang.srt.utils.common import SafeUnpickler

        entries = sorted(
            (m, n)
            for (m, n) in SafeUnpickler.ALLOWED_GLOBALS
            if m.startswith("sglang.") and (m, n) not in self._NPU_ONLY
        )
        self.assertTrue(entries)
        for module, name in entries:
            with self.subTest(module=module, name=name):
                mod = importlib.import_module(module)
                self.assertTrue(hasattr(mod, name), f"{module}.{name} missing")

    def test_round_trips_embedding_data_from_fork_module(self):
        from sglang.srt.disaggregation.encode_receiver import EmbeddingData
        from sglang.srt.managers.schedule_batch import Modality

        value = EmbeddingData(
            req_id="r0",
            num_parts=1,
            part_idx=0,
            grid_dim=[1, 2, 2],
            modality=Modality.IMAGE,
            embedding=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        )
        restored = safe_pickle_loads(
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        )
        self.assertIsInstance(restored, EmbeddingData)
        self.assertEqual(restored.modality, Modality.IMAGE)
        self.assertEqual(restored.grid_dim, [1, 2, 2])
        self.assertTrue(torch.equal(restored.embedding, value.embedding))

    def test_rejects_sglang_code_module_gadget(self):
        # The old prefix list allowed every global under sglang.srt.utils.*,
        # including dynamic_import (an import-by-string gadget).
        with self.assertRaisesRegex(
            RuntimeError,
            r"Blocked unsafe global \(sglang\.srt\.utils\.common\.dynamic_import\)",
        ):
            safe_pickle_loads(b"csglang.srt.utils.common\ndynamic_import\n.")


if __name__ == "__main__":
    unittest.main()
