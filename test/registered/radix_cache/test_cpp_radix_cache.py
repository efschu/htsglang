import unittest

from flliper.srt.environ import envs
from flliper.srt.utils import kill_process_tree
from flliper.test.ci.ci_register import register_cuda_ci
from flliper.test.kits.eval_accuracy_kit import MMLUMixin
from flliper.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Note: AMD registration removed - test_cpp_radix_cache fails on AMD due to C++ radix tree issues
register_cuda_ci(est_time=60, suite="nightly-1-gpu", nightly=True)


class TestCppRadixCache(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        envs.FLLIPER_EXPERIMENTAL_CPP_RADIX_TREE.set(True)
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
