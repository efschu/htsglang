"""`--enable-memory-saver` must reach the hybrid-SWA KV sub-pools.

`SWAKVPool` allocates nothing itself: the two sub-pools it composes are the
allocation. It used to overwrite `kwargs["enable_memory_saver"] = False`
unconditionally, so on any hybrid-SWA model the flag was silently a no-op --
the pool that memory saver exists to release was the one pool it could not
touch. The unified-buffer replacement (`UnifiedSWAKVPool`) already honoured
the setting, which is why the gap survived: only the legacy static-partition
path had it.
"""

import ast
import pathlib
import unittest
from unittest.mock import patch

import torch

import sglang.srt.model_executor.model_runner_kv_cache_mixin as kv_cache_mixin
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RecordingKVPool:
    """Stands in for the MHA sub-pool and records the kwargs it was handed."""

    calls = []

    def __init__(self, **kwargs):
        RecordingKVPool.calls.append(kwargs)

    def get_kv_size_bytes(self):
        return 0, 0


class TestSWAKVPoolMemorySaver(CustomTestCase):
    def setUp(self):
        RecordingKVPool.calls = []

    def _build_pool(self, enable_memory_saver=None):
        kwargs = {}
        if enable_memory_saver is not None:
            kwargs["enable_memory_saver"] = enable_memory_saver
        with patch(
            "sglang.srt.mem_cache.swa_memory_pool.maybe_init_custom_mem_pool",
            return_value=(False, None, None),
        ):
            return SWAKVPool(
                size=8,
                size_swa=4,
                page_size=1,
                dtype=torch.float16,
                head_num=2,
                head_dim=4,
                swa_attention_layer_ids=[0],
                full_attention_layer_ids=[1],
                device="cpu",
                token_to_kv_pool_class=RecordingKVPool,
                **kwargs,
            )

    def test_forwards_memory_saver_setting_to_both_subpools(self):
        self._build_pool(True)
        self.assertEqual(
            [call["enable_memory_saver"] for call in RecordingKVPool.calls],
            [True, True],
        )

    def test_defaults_memory_saver_off_for_existing_callers(self):
        self._build_pool()
        self.assertEqual(
            [call["enable_memory_saver"] for call in RecordingKVPool.calls],
            [False, False],
        )

    def test_every_swakvpool_callsite_passes_the_server_setting(self):
        """Forwarding is only worth anything if the callers actually pass it.

        Constructing the model runner here would mean mocking most of it, so
        this asserts the structural invariant instead: every `SWAKVPool(...)`
        in the KV-cache mixin names `enable_memory_saver`. A new pool branch
        added without it fails here rather than at someone's next suspend.
        """
        source = pathlib.Path(kv_cache_mixin.__file__).read_text()
        tree = ast.parse(source)

        callsites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SWAKVPool"
        ]
        self.assertTrue(callsites, "no SWAKVPool construction found -- test is stale")

        for call in callsites:
            with self.subTest(line=call.lineno):
                names = {kw.arg for kw in call.keywords}
                self.assertIn(
                    "enable_memory_saver",
                    names,
                    f"SWAKVPool at line {call.lineno} does not forward "
                    "enable_memory_saver",
                )


if __name__ == "__main__":
    unittest.main()
