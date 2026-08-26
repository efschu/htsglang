"""#894 S5 -- ``SGLANG_GGUF_MMQ_DECODE_THRESHOLD`` beats the CLI flag, silently.

THE DEFECT, at base commit 2b13ba92d1 (= pin 0cd27d957d + #889)
---------------------------------------------------------------
``_mmq_decode_threshold_enabled`` (``gguf.py:600-621``) short-circuits on
PRESENCE, not on value::

    env = os.environ.get(_MMQ_THRESHOLD_ENV)
    if env is not None:
        _mmq_threshold_cached = env == "1"
        return _mmq_threshold_cached

So a stale ``SGLANG_GGUF_MMQ_DECODE_THRESHOLD=0`` left in a shell or an env
capture from an old A/B run beats ``--gguf-mmq-decode-threshold`` without ever
consulting it, and never says a word: the ONLY log on the whole path
(``gguf.py:682-694``, "MMQ decode threshold ACTIVE") fires when a reroute
actually happens, so the losing configuration produces zero reroutes and
therefore zero output. Flag on, nothing rerouted, nothing logged --
structurally the same shape as #889's inert PP window.

The precedence itself is DOCUMENTED and stays: the CLI help says the env var
"wins over this flag", and the override exists so an A/B run can flip the
kernel choice without re-parsing ServerArgs. This changes nothing about who
wins. It ends the silence about it.

WARNING, NOT REFUSAL, decided on the danger direction
-----------------------------------------------------
* This runs inside the GGUF matmul dispatch, on the first quantized matmul of
  the forward pass. Raising there is not a parse-time refusal, it is a model
  that dies mid-forward -- on any process that carries a stale env var, which
  is exactly the population the fix is for.
* The defect's blast radius is a wrong belief about which kernel ran (and a
  benchmark attributed to the wrong arm). A refusal's blast radius is the
  instance.
* Nor may the code flip the precedence to "flag wins": that would break the
  documented, deliberately restart-free override the A/B harness uses, and it
  would move a KERNEL SELECTION -- MMQ and MMVQ are not bit-identical -- as a
  side effect of a logging fix. State the truth; do not change it.

A SECOND SILENCE IN THE SAME THREE LINES
----------------------------------------
``env == "1"`` means every other spelling -- ``"true"``, ``"yes"``, ``"on"``,
``"01"`` -- is read as OFF without complaint. Same class, same fix: say so.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest
from unittest import mock

from sglang.srt.layers.quantization import gguf as G
from sglang.test.test_utils import CustomTestCase

ENV = "SGLANG_GGUF_MMQ_DECODE_THRESHOLD"
LOGGER = "sglang.srt.layers.quantization.gguf"


class _Base(CustomTestCase):
    def setUp(self):
        G._reset_mmq_threshold_cache()
        self.addCleanup(G._reset_mmq_threshold_cache)

    def _decide(self, env, flag, *, published=True):
        """Resolve the enable decision, returning (value, [log lines])."""
        patches = []
        if env is None:
            patches.append(mock.patch.dict(G.os.environ, {}, clear=False))
            G.os.environ.pop(ENV, None)
        else:
            patches.append(mock.patch.dict(G.os.environ, {ENV: env}))
        if published:
            args = types.SimpleNamespace(gguf_mmq_decode_threshold=flag)
            patches.append(
                mock.patch(
                    "sglang.srt.runtime_context.get_server_args", return_value=args
                )
            )
        else:
            patches.append(
                mock.patch(
                    "sglang.srt.runtime_context.get_server_args",
                    side_effect=RuntimeError("ServerArgs not published yet"),
                )
            )
        with self._maybe_logs() as captured:
            for p in patches:
                p.start()
            try:
                value = G._mmq_decode_threshold_enabled()
            finally:
                for p in reversed(patches):
                    p.stop()
        return value, captured.output

    def _maybe_logs(self):
        # assertLogs fails when nothing is logged, which is itself one of the
        # assertions here, so capture at a level that always has a record.
        return _Capture(self, LOGGER)


class _Capture:
    def __init__(self, case, logger_name):
        self._case = case
        self._name = logger_name
        self.output = []

    def __enter__(self):
        import logging

        self._records = []
        self._handler = _ListHandler(self._records)
        self._logger = logging.getLogger(self._name)
        self._prev_level = self._logger.level
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prev_level)
        self.output.extend(self._records)
        return False


class _ListHandler:
    """Minimal handler: no formatting, no level filtering, no side effects."""

    level = 0

    def __init__(self, sink):
        self._sink = sink

    def handle(self, record):
        self._sink.append(f"{record.levelname}:{record.getMessage()}")

    def acquire(self):
        pass

    def release(self):
        pass

    def createLock(self):
        pass


class TestTheEnvOverrideAnnouncesItself(_Base):
    def test_stale_env_zero_beating_the_flag_is_announced(self):
        """RED AT BASE. The exact stale-A/B shape: flag on, env off, silence."""
        value, logs = self._decide("0", True)
        self.assertFalse(value)
        warnings = [line for line in logs if line.startswith("WARNING")]
        self.assertEqual(
            len(warnings),
            1,
            f"expected exactly one warning, got {logs!r}",
        )
        line = warnings[0]
        self.assertIn(ENV, line)
        self.assertIn("--gguf-mmq-decode-threshold", line)
        # Both sides of the disagreement have to be legible, or the reader
        # cannot tell which one they set.
        self.assertIn("'0'", line)
        self.assertIn("#894", line)

    def test_env_one_beating_a_flag_that_is_off_is_announced_too(self):
        """The other direction is equally a surprise: a kernel change nobody
        asked for on this command line."""
        value, logs = self._decide("1", False)
        self.assertTrue(value)
        self.assertEqual(len([x for x in logs if x.startswith("WARNING")]), 1)

    def test_agreement_is_not_warned_about(self):
        """No noise on configurations that were never ambiguous."""
        for env, flag in (("1", True), ("0", False)):
            with self.subTest(env=env, flag=flag):
                G._reset_mmq_threshold_cache()
                _, logs = self._decide(env, flag)
                self.assertEqual([x for x in logs if x.startswith("WARNING")], [])

    def test_no_env_is_the_untouched_default_path(self):
        for flag in (False, True):
            with self.subTest(flag=flag):
                G._reset_mmq_threshold_cache()
                value, logs = self._decide(None, flag)
                self.assertEqual(value, flag)
                self.assertEqual(logs, [])

    def test_an_unparseable_value_says_it_is_being_read_as_off(self):
        """RED AT BASE: ``env == "1"`` reads 'true' as OFF and says nothing."""
        value, logs = self._decide("true", True)
        self.assertFalse(value)
        warnings = [x for x in logs if x.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("'true'", warnings[0])
        self.assertIn("0|1", warnings[0])

    def test_an_unparseable_value_warns_even_when_the_flag_agrees(self):
        """A typo is a defect on its own, independent of who wins."""
        value, logs = self._decide("yes", False)
        self.assertFalse(value)
        self.assertEqual(len([x for x in logs if x.startswith("WARNING")]), 1)

    def test_it_still_speaks_when_server_args_are_not_published(self):
        """The comparison is impossible, the supersession is not.

        Standalone kernel use and unit tests reach this before ModelRunner
        publishes ServerArgs. The env still governs, and saying so is the whole
        point -- reporting nothing here would leave the same silence for the
        callers most likely to be surprised by it.
        """
        value, logs = self._decide("0", True, published=False)
        self.assertFalse(value)
        warnings = [x for x in logs if x.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)
        self.assertIn(ENV, warnings[0])

    def test_the_warning_is_latched_to_one_line_per_process(self):
        value, logs = self._decide("0", True)
        self.assertFalse(value)
        self.assertEqual(len([x for x in logs if x.startswith("WARNING")]), 1)
        # The decision is cached, so repeated dispatch must not re-log.
        with self._maybe_logs() as second:
            for _ in range(5):
                G._mmq_decode_threshold_enabled()
        self.assertEqual(second.output, [])

    def test_the_reset_hook_clears_the_warning_latch(self):
        """Otherwise the hook leaves a test-order-dependent gap in this suite."""
        self._decide("0", True)
        G._reset_mmq_threshold_cache()
        _, logs = self._decide("0", True)
        self.assertEqual(len([x for x in logs if x.startswith("WARNING")]), 1)

    def test_precedence_is_unchanged_in_every_combination(self):
        """The fix announces the winner; it must not change one."""
        for env, flag, expected in (
            ("0", True, False),
            ("0", False, False),
            ("1", True, True),
            ("1", False, True),
            ("true", True, False),
            (None, True, True),
            (None, False, False),
        ):
            with self.subTest(env=env, flag=flag):
                G._reset_mmq_threshold_cache()
                value, _ = self._decide(env, flag)
                self.assertIs(value, expected)


if __name__ == "__main__":
    unittest.main()
