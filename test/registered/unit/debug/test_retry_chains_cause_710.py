"""retry() must chain the cause, or every environmental failure reads as noise.

Found during the mem_cache triage (#585). ``CustomTestCase._callTestMethod``
wraps every test in ``sglang.srt.utils.common.retry``, and that helper re-raised
a bare ``Exception("retry() exceed maximum number of retries.")`` with no
``from e``. The original error was discarded, so 841 of 944 hermetic mem_cache
failures reported an opaque retry message instead of the one-line reason
("No accelerator ... is available"). One environmental cause looked like a
suite-wide catastrophe, and the triage had to reconstruct the cause by hand.

This is not mem_cache-specific: the wrapper is on EVERY test in the corpus, so
any suite hitting a deterministic failure inherits the same blindfold. Chaining
costs two words and returns the reason to every future traceback.

Both raise sites are covered -- the retries-exhausted one and the
should_retry-refused one -- because a fix that chained only the first would
still discard the cause on the other path.

Hermetic: pure Python, no CUDA.
"""

import pytest

from sglang.srt.utils.common import retry


class _Distinctive(RuntimeError):
    pass


def test_exhausted_retries_chain_the_original_cause():
    def always_fails():
        raise _Distinctive("the real reason")

    with pytest.raises(Exception) as excinfo:
        retry(always_fails, max_retry=1, initial_delay=0.0, max_delay=0.0)

    assert "exceed maximum number of retries" in str(excinfo.value)
    cause = excinfo.value.__cause__
    assert isinstance(cause, _Distinctive), f"cause was {cause!r}, not the original"
    assert "the real reason" in str(cause)


def test_a_refused_retry_also_chains_the_cause():
    def always_fails():
        raise _Distinctive("refused reason")

    with pytest.raises(Exception) as excinfo:
        retry(
            always_fails,
            max_retry=5,
            initial_delay=0.0,
            max_delay=0.0,
            should_retry=lambda e: False,
        )

    assert "should not be retried" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, _Distinctive)
    assert "refused reason" in str(excinfo.value.__cause__)


def test_the_cause_reaches_a_formatted_traceback():
    """The point is what an operator READS, so assert on the rendered text."""
    import traceback

    def always_fails():
        raise _Distinctive("visible in traceback")

    try:
        retry(always_fails, max_retry=0, initial_delay=0.0, max_delay=0.0)
    except Exception as exc:
        rendered = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    assert "visible in traceback" in rendered
    assert "direct cause" in rendered


def test_a_successful_call_is_unaffected():
    calls = []

    def succeeds_second_time():
        calls.append(1)
        if len(calls) < 2:
            raise _Distinctive("transient")
        return "ok"

    assert (
        retry(succeeds_second_time, max_retry=3, initial_delay=0.0, max_delay=0.0)
        == "ok"
    )
