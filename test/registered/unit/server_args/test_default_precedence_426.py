"""#426 -- a model default yields to a feature default; user input never does.

Upstream sgl-project/sglang#33199: with ``DeepseekV4ForCausalLM`` and
speculative decoding enabled but ``--max-running-requests`` unset, the
effective value stayed at 256 instead of the documented 48. Nothing is
mis-parsed; the cause is resolution ORDER. The model hook runs first and fills
``max_running_requests = 256``; the speculative hook runs later and guards with
``is None``; the field is no longer None, so the feature default never applies.
The model hook's own comment even asserts that the speculative hook is the
later writer of that field.

``is None`` is a proxy for "the user did not set this", and it stops being true
the moment an earlier hook writes. Our #379 fixed a sibling instance by giving
"off" exactly one representation instead of trusting a downstream ``is None``;
#426 adopts upstream's generalization as a helper
(``arg_groups/default_precedence.py``) so the distinction is provenance, not
emptiness:

    unset        -- either kind may claim it
    model default -- a feature default overwrites it
    feature default -- later defaults of either kind leave it alone
    user value   -- untouchable

The behavior change is exactly one combination: DeepSeek-V4 + speculative
decoding + no explicit flag. Every other combination is pinned unchanged here,
because "byte-identical everywhere else" is the whole compatibility claim.

GPU-free: argument resolution only.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import sglang.srt.arg_groups.speculative_hook as speculative_hook
from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DSV4_ARCH = "DeepseekV4ForCausalLM"
DSV4_MAX_RUNNING_REQUESTS = 256
SPECULATIVE_MAX_RUNNING_REQUESTS = 48

MODEL_DEFAULT = "model"
FEATURE_DEFAULT = "feature"


def _precedence():
    """The helper module, or None on a tree that does not have it yet.

    Everything below is resolved lazily and by name so the behavioral
    assertions fail on their own terms against an unfixed tree instead of
    erroring at import: a missing module is not evidence that a default
    resolved wrongly.
    """
    try:
        from sglang.srt.arg_groups import default_precedence
    except ImportError:  # pragma: no cover - unfixed tree only
        return None
    return default_precedence


def default_provenance(server_args, field):
    module = _precedence()
    return None if module is None else module.default_provenance(server_args, field)


def set_model_default(server_args, field, value):
    module = _precedence()
    if module is not None:
        return module.set_model_default(server_args, field, value)
    if getattr(server_args, field) is not None:  # pragma: no cover
        return False
    setattr(server_args, field, value)
    return True


def set_feature_default(server_args, field, value):
    module = _precedence()
    if module is not None:
        return module.set_feature_default(server_args, field, value)
    if getattr(server_args, field) is not None:  # pragma: no cover
        return False
    setattr(server_args, field, value)
    return True


def _apply_speculative_fill(server_args):
    """Run the speculative-decoding fill the way the algorithm handlers do.

    On an unfixed tree the helper does not exist, so this reproduces the
    handlers' legacy body verbatim -- the ``is None`` guard that is the bug.
    """
    fill = getattr(speculative_hook, "_set_speculative_max_running_requests", None)
    if fill is not None:
        return fill(server_args)
    if server_args.max_running_requests is None:  # pragma: no cover
        server_args.max_running_requests = SPECULATIVE_MAX_RUNNING_REQUESTS


def _server_args(max_running_requests=None, arch=DSV4_ARCH):
    """The minimum surface the two hooks read. A real ServerArgs would drag in
    a model download; the hooks touch only these fields."""
    hf_config = SimpleNamespace(architectures=[arch])
    model_config = SimpleNamespace(hf_config=hf_config)
    return SimpleNamespace(
        max_running_requests=max_running_requests,
        speculative_algorithm=None,
        kv_cache_dtype="auto",
        device="cuda",
        _resolved_overrides=[],
        get_model_config=lambda: model_config,
    )


class TestTheReportedCombination(CustomTestCase):
    """The falsifier: unfixed, the model's 256 survives the speculative hook."""

    def test_speculative_decoding_reaches_its_documented_default(self):
        server_args = _server_args()
        apply_deepseek_v4_defaults(server_args, DSV4_ARCH)
        self.assertEqual(server_args.max_running_requests, DSV4_MAX_RUNNING_REQUESTS)
        self.assertEqual(
            default_provenance(server_args, "max_running_requests"), MODEL_DEFAULT
        )

        _apply_speculative_fill(server_args)
        self.assertEqual(
            server_args.max_running_requests, SPECULATIVE_MAX_RUNNING_REQUESTS
        )
        self.assertEqual(
            default_provenance(server_args, "max_running_requests"), FEATURE_DEFAULT
        )


class TestEveryOtherCombinationIsUnchanged(CustomTestCase):
    def test_explicit_user_input_beats_both_hooks(self):
        server_args = _server_args(max_running_requests=128)
        apply_deepseek_v4_defaults(server_args, DSV4_ARCH)
        _apply_speculative_fill(server_args)
        self.assertEqual(server_args.max_running_requests, 128)
        self.assertIsNone(default_provenance(server_args, "max_running_requests"))

    def test_explicit_user_input_equal_to_the_model_default_still_wins(self):
        """A user who types 256 must keep 256, even though it is numerically
        the value the model hook would have written -- this is precisely the
        case an ``is None`` guard cannot tell apart."""
        server_args = _server_args(max_running_requests=DSV4_MAX_RUNNING_REQUESTS)
        apply_deepseek_v4_defaults(server_args, DSV4_ARCH)
        _apply_speculative_fill(server_args)
        self.assertEqual(server_args.max_running_requests, DSV4_MAX_RUNNING_REQUESTS)

    def test_the_model_default_stands_without_speculative_decoding(self):
        server_args = _server_args()
        apply_deepseek_v4_defaults(server_args, DSV4_ARCH)
        self.assertEqual(server_args.max_running_requests, DSV4_MAX_RUNNING_REQUESTS)

    def test_speculative_decoding_alone_is_unaffected(self):
        """No model hook ran, so the old ``is None`` path and the new one must
        agree."""
        server_args = _server_args()
        _apply_speculative_fill(server_args)
        self.assertEqual(
            server_args.max_running_requests, SPECULATIVE_MAX_RUNNING_REQUESTS
        )

    def test_two_speculative_handlers_do_not_fight(self):
        """Several algorithm handlers fill the same field; the first write must
        remain the observable one, as it was under ``is None``."""
        server_args = _server_args()
        _apply_speculative_fill(server_args)
        _apply_speculative_fill(server_args)
        self.assertEqual(
            server_args.max_running_requests, SPECULATIVE_MAX_RUNNING_REQUESTS
        )


class TestThePrecedenceHelper(CustomTestCase):
    def test_a_model_default_does_not_overwrite_a_feature_default(self):
        """The rule is directional. Reversing the hook order must not reverse
        the outcome."""
        server_args = SimpleNamespace(field=None)
        self.assertTrue(set_feature_default(server_args, "field", 48))
        self.assertFalse(set_model_default(server_args, "field", 256))
        self.assertEqual(server_args.field, 48)

    def test_a_model_default_does_not_overwrite_a_user_value(self):
        server_args = SimpleNamespace(field=7)
        self.assertFalse(set_model_default(server_args, "field", 256))
        self.assertEqual(server_args.field, 7)

    def test_provenance_is_per_field(self):
        server_args = SimpleNamespace(a=None, b=None)
        set_model_default(server_args, "a", 1)
        set_feature_default(server_args, "b", 2)
        self.assertEqual(default_provenance(server_args, "a"), MODEL_DEFAULT)
        self.assertEqual(default_provenance(server_args, "b"), FEATURE_DEFAULT)
        self.assertIsNone(default_provenance(server_args, "c"))

    def test_an_untouched_object_reports_nothing(self):
        self.assertIsNone(default_provenance(SimpleNamespace(), "anything"))


class TestNoHandlerKeepsTheOldGuard(CustomTestCase):
    """Ratchet: the generalization is only real if every filler uses it."""

    def test_speculative_hook_has_no_bare_is_none_fill_left(self):
        source = Path(speculative_hook.__file__).read_text()
        stale = re.findall(
            r"server_args\.max_running_requests\s*=\s*\d+",
            source,
        )
        self.assertEqual(stale, [], f"bare max_running_requests fills: {stale}")

    def test_every_algorithm_handler_routes_through_the_helper(self):
        source = Path(speculative_hook.__file__).read_text()
        calls = source.count("_set_speculative_max_running_requests(server_args)")
        self.assertEqual(
            calls,
            5,
            "the five algorithm handlers (EAGLE family, DSPARK, NGRAM, "
            "frozen-KV MTP, DFLASH) must all fill through one place",
        )


if __name__ == "__main__":
    unittest.main()
