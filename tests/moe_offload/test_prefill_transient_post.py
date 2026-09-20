"""Task #48 C: the measured prefill transient as an explicit KV-budget post."""
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    PREFILL_TRANSIENT_ENV,
    prefill_transient_mib_for_rank,
)


def test_scalar_applies_to_every_rank():
    assert prefill_transient_mib_for_rank("2240", 0) == 2240.0
    assert prefill_transient_mib_for_rank("2240", 2) == 2240.0


def test_vector_is_per_rank_and_short_vectors_book_nothing():
    assert prefill_transient_mib_for_rank("2240,1800,1800", 1) == 1800.0
    assert prefill_transient_mib_for_rank("2240,1800", 2) == 0.0


def test_empty_or_garbage_books_nothing():
    assert prefill_transient_mib_for_rank("", 0) == 0.0
    assert prefill_transient_mib_for_rank("abc", 0) == 0.0
    assert prefill_transient_mib_for_rank("-5", 0) == 0.0
    assert PREFILL_TRANSIENT_ENV == "SGLANG_KV_BUDGET_PREFILL_TRANSIENT_MIB"
