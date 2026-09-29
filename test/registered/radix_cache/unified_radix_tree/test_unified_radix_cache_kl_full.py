import unittest

from flliper.srt.utils import kill_process_tree
from flliper.test.ci.ci_register import register_amd_ci, register_cuda_ci
from flliper.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from flliper.test.kl_multiturn_utils import get_input_ids
from flliper.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=250, stage="base-b", runner_config="2-gpu-large")
register_amd_ci(est_time=400, suite="stage-b-test-2-gpu-large-amd")

FULL_MODEL = "Qwen/Qwen3-32B"


class TestUnifiedFullRadixCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Full attention."""

    kl_threshold = 0.0025

    @classmethod
    def setUpClass(cls):
        cls.model = FULL_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.80",
                "--page-size",
                "64",
            ],
            env={"FLLIPER_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
