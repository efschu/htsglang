import unittest

from flliper.srt.utils import kill_process_tree
from flliper.test.ci.ci_register import register_cuda_ci
from flliper.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from flliper.test.kl_multiturn_utils import get_input_ids
from flliper.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=250, stage="base-b", runner_config="2-gpu-large")

SWA_MODEL = "openai/gpt-oss-20b"


class TestUnifiedSWARadixCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """SWA hybrid + UnifiedRadixCache."""

    kl_threshold = 0.03
    gsm8k_threshold = 0.7
    mmlu_threshold = 0.7

    @unittest.skipIf(is_in_ci(), "SWA model mmlu eval not stable enough")
    def test_mmlu(self):
        super().test_mmlu()

    @classmethod
    def setUpClass(cls):
        cls.model = SWA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.7",
                "--disable-piecewise-cuda-graph",
            ],
            env={"FLLIPER_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
