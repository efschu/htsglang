# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The pin path's ledger must not carry a zero for its largest post (#605).

RED-FIRST, and the red is the point: the completeness check must FAIL on the
ledger the pin path actually produced on boot 1464299, where every card's
``model weights (shards)`` term reads 0 MiB while the boot loaded 13674 / 8325
/ 9293 MiB of them. A zero there is not a small error -- it is the dominant
post of the whole ledger, and it looked exactly like a priced term.
"""

import unittest

from sglang.srt.mem_ledger.engine import (
    TERM_HARDWARE_RESIDUAL,
    TERM_NVML_CARVE_OUT,
    TERM_WEIGHTS,
)
from sglang.srt.mem_ledger.reconcile import (
    LedgerIncomplete,
    completeness_failures,
    require_complete,
)


def _card(terms, gpu_id=0):
    return {
        "gpu_id": gpu_id,
        "card": f"GPU {gpu_id} (NVIDIA GeForce RTX 5090, NVML total 32607 MiB)",
        "ranks": [gpu_id],
        "demand_mib": sum(m for _n, m in terms),
        "kv_pool_mib": 29927,
        "unbounded": [],
        "terms": [{"name": n, "mib": m, "provenance": "modeled"} for n, m in terms],
    }


#: The real shape of ledger_1464299-1786612548.json, one card.
PIN_PATH_LEDGER = {
    "boot_id": "1464299-1786612548",
    "cards": [
        _card(
            [
                (TERM_WEIGHTS, 0),
                ("load transient (allocator peak over resident)", 70),
                ("GDN prefill scratch", 20),
                ("attention workspaces (capped)", 384),
                ("NCCL communicator buffers", 0),
                (TERM_HARDWARE_RESIDUAL, 664),
                (TERM_NVML_CARVE_OUT, 518),
            ]
        )
    ],
}


class TestTheZeroWeightLedgerIsIncomplete(unittest.TestCase):
    def test_the_shipped_pin_path_ledger_fails_the_completeness_check(self):
        failures = completeness_failures(PIN_PATH_LEDGER)
        self.assertTrue(failures)
        joined = " ".join(failures)
        self.assertIn(TERM_WEIGHTS, joined)
        self.assertIn("GPU 0", joined)

    def test_require_complete_raises_on_it(self):
        with self.assertRaises(LedgerIncomplete) as ctx:
            require_complete(PIN_PATH_LEDGER)
        self.assertIn(TERM_WEIGHTS, str(ctx.exception))

    def test_the_failure_says_WHY_a_zero_here_is_not_a_price(self):
        """A reader has to learn that this is the pin path skipping the
        planner, not a model with no weights."""
        message = " ".join(completeness_failures(PIN_PATH_LEDGER))
        self.assertIn("pin path", message.lower())


class TestAPricedWeightTermPasses(unittest.TestCase):
    def test_a_ledger_with_real_weights_is_complete(self):
        ledger = {
            "cards": [
                _card(
                    [
                        (TERM_WEIGHTS, 13850),
                        (TERM_HARDWARE_RESIDUAL, 902),
                        (TERM_NVML_CARVE_OUT, 518),
                    ]
                )
            ]
        }
        self.assertEqual(completeness_failures(ledger), [])
        require_complete(ledger)

    def test_a_REFUSED_weight_term_is_complete_because_it_is_honest(self):
        """A refusal is a statement; a zero is a false price. The check
        accepts the first and rejects the second."""
        card = _card([(TERM_HARDWARE_RESIDUAL, 902), (TERM_NVML_CARVE_OUT, 518)])
        card["unbounded"] = [
            f"{TERM_WEIGHTS} on NVIDIA GeForce RTX 5090: the pipeline-stage "
            "weight split is not modelled"
        ]
        self.assertEqual(completeness_failures({"cards": [card]}), [])

    def test_a_missing_weight_term_entirely_is_also_incomplete(self):
        card = _card([(TERM_HARDWARE_RESIDUAL, 902), (TERM_NVML_CARVE_OUT, 518)])
        failures = completeness_failures({"cards": [card]})
        self.assertTrue(failures)
        self.assertIn(TERM_WEIGHTS, " ".join(failures))


class TestEveryCardIsChecked(unittest.TestCase):
    def test_a_second_card_with_a_zero_is_named_too(self):
        ledger = {
            "cards": [
                _card([(TERM_WEIGHTS, 13850), (TERM_NVML_CARVE_OUT, 518)], gpu_id=0),
                _card([(TERM_WEIGHTS, 0), (TERM_NVML_CARVE_OUT, 425)], gpu_id=1),
            ]
        }
        failures = completeness_failures(ledger)
        self.assertEqual(len(failures), 1)
        self.assertIn("GPU 1", failures[0])


if __name__ == "__main__":
    unittest.main()
