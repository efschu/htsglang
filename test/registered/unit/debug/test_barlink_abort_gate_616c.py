# SPDX-License-Identifier: Apache-2.0
"""Hermetic unit tests for barlink_abort_gate -- no CUDA, no torch device work.

Every test exercises pure-python logic: env-var parsing, predicate functions,
threshold clamping, registry CRUD, capture-depth context manager, and the
decision helpers.  Nothing here touches a GPU or imports torch.
"""

from __future__ import annotations


import pytest

from sglang.srt.distributed.device_communicators.barlink_abort_gate import (
    ENV_DEFER,
    ENV_ENABLE,
    ENV_EVERY,
    ENV_MAX_LAG,
    ENV_POLL_MS,
    ENV_REPLAY,
    abort_check_enabled,
    check_aborts,
    check_after_graph_replay,
    check_every,
    defer_enabled,
    max_lag,
    pause_polling,
    poll_interval_s,
    poll_status_words,
    register,
    registered,
    replay_check_enabled,
    reset_for_test,
    should_defer_status,
    should_poll_status,
    unregister,
)


# ---------------------------------------------------------------------------
# _env_flag (indirect via abort_check_enabled, defer_enabled, etc.)
# ---------------------------------------------------------------------------


class TestEnvFlagDefaults:
    """_env_flag returns default when the variable is unset (line 156-158)."""

    def test_enable_default_true_when_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_ENABLE, raising=False)
        assert abort_check_enabled() is True

    def test_defer_default_true_when_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_DEFER, raising=False)
        assert defer_enabled() is True

    def test_replay_default_true_when_unset(self, monkeypatch):
        # replay_check_enabled is AND of enable + replay; set enable explicitly.
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.delenv(ENV_REPLAY, raising=False)
        assert replay_check_enabled() is True


class TestEnvFlagTruthyFalsy:
    """_env_flag parses truthy/falsy values (line 159).

    _FALSE tuple defined at line 149: ("0", "false", "no", "off", "")
    The code does: raw.strip().lower() not in _FALSE
    So anything NOT in _FALSE is truthy.
    """

    # Values in _FALSE that must yield False.
    @pytest.mark.parametrize(
        "val", ["0", "false", "False", "FALSE", "no", "No", "off", "Off", "OFF", ""]
    )
    def test_falsy_values(self, val, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, val)
        assert abort_check_enabled() is False

    # Values NOT in _FALSE that must yield True.
    @pytest.mark.parametrize(
        "val",
        [
            "1",
            "true",
            "True",
            "TRUE",
            "yes",
            "on",
            "anything",
            "garbage",
            "2",
            "random",
        ],
    )
    def test_truthy_values(self, val, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, val)
        assert abort_check_enabled() is True


class TestEnvFlagUnrecognisedValue:
    """Unrecognised / non-standard strings behave as the code says.

    Line 159: `raw.strip().lower() not in _FALSE` -- this means ANY string not
    in _FALSE ("0","false","no","off","") is treated as truthy.  There is no
    "raise on unknown value" path.
    """

    def test_random_string_is_truthy(self, monkeypatch):
        """A value like 'xyz' is NOT in _FALSE, so the result is True."""
        monkeypatch.setenv(ENV_ENABLE, "xyz")
        assert abort_check_enabled() is True

    def test_whitespace_stripped_then_checked(self, monkeypatch):
        """'  false  ' strips to 'false' which IS in _FALSE -> False."""
        monkeypatch.setenv(ENV_ENABLE, "  false  ")
        assert abort_check_enabled() is False

    def test_whitespace_around_truthy_value(self, monkeypatch):
        """'  TRUE  ' strips to 'true' which is NOT in _FALSE -> True."""
        monkeypatch.setenv(ENV_ENABLE, "  TRUE  ")
        assert abort_check_enabled() is True


# ---------------------------------------------------------------------------
# check_every -- integer clamping (lines 171-179)
# ---------------------------------------------------------------------------


class TestCheckEvery:
    def test_default_is_one(self, monkeypatch):
        monkeypatch.delenv(ENV_EVERY, raising=False)
        assert check_every() == 1

    def test_positive_value(self, monkeypatch):
        monkeypatch.setenv(ENV_EVERY, "5")
        assert check_every() == 5

    def test_zero_clamped_to_one(self, monkeypatch):
        """Values <= 0 read as 1 (line 176: max(int(raw), 1))."""
        monkeypatch.setenv(ENV_EVERY, "0")
        assert check_every() == 1

    def test_negative_clamped_to_one(self, monkeypatch):
        monkeypatch.setenv(ENV_EVERY, "-3")
        assert check_every() == 1

    def test_non_integer_falls_back_to_one(self, monkeypatch, caplog):
        """ValueError -> warning log + fallback 1 (lines 177-179)."""
        monkeypatch.setenv(ENV_EVERY, "abc")
        assert check_every() == 1
        assert "is not an integer" in caplog.text


# ---------------------------------------------------------------------------
# max_lag -- integer clamping (lines 187-196)
# ---------------------------------------------------------------------------


class TestMaxLag:
    def test_default_is_four(self, monkeypatch):
        monkeypatch.delenv(ENV_MAX_LAG, raising=False)
        assert max_lag() == 4

    def test_positive_value(self, monkeypatch):
        monkeypatch.setenv(ENV_MAX_LAG, "10")
        assert max_lag() == 10

    def test_zero_clamped_to_one(self, monkeypatch):
        monkeypatch.setenv(ENV_MAX_LAG, "0")
        assert max_lag() == 1

    def test_negative_clamped_to_one(self, monkeypatch):
        monkeypatch.setenv(ENV_MAX_LAG, "-1")
        assert max_lag() == 1

    def test_non_integer_falls_back_to_four(self, monkeypatch, caplog):
        """ValueError -> warning log + fallback 4 (lines 194-196)."""
        monkeypatch.setenv(ENV_MAX_LAG, "notanint")
        assert max_lag() == 4
        assert "is not an integer" in caplog.text


# ---------------------------------------------------------------------------
# poll_interval_s -- float parsing (lines 199-208)
# ---------------------------------------------------------------------------


class TestPollIntervalS:
    def test_default_10ms(self, monkeypatch):
        monkeypatch.delenv(ENV_POLL_MS, raising=False)
        assert poll_interval_s() == 0.010

    def test_explicit_ms(self, monkeypatch):
        monkeypatch.setenv(ENV_POLL_MS, "50")
        assert poll_interval_s() == 0.050  # 50 ms / 1000

    def test_zero_converted_to_seconds(self, monkeypatch):
        monkeypatch.setenv(ENV_POLL_MS, "0")
        assert poll_interval_s() == 0.0

    def test_negative_clamped_to_zero(self, monkeypatch):
        """max(float(raw), 0.0) at line 205."""
        monkeypatch.setenv(ENV_POLL_MS, "-10")
        assert poll_interval_s() == 0.0

    def test_non_numeric_falls_back_to_10ms(self, monkeypatch, caplog):
        monkeypatch.setenv(ENV_POLL_MS, "abc")
        assert poll_interval_s() == 0.010
        assert "is not a number" in caplog.text


# ---------------------------------------------------------------------------
# should_poll_status and should_defer_status -- pure predicates
# ---------------------------------------------------------------------------


class TestShouldPollStatus:
    """Lines 287-300: returns bool(status_is_cuda) and bool(watchdog_running)."""

    def test_both_true(self):
        assert should_poll_status(True, True) is True

    def test_cuda_false(self):
        assert should_poll_status(False, True) is False

    def test_watchdog_false(self):
        assert should_poll_status(True, False) is False

    def test_both_false(self):
        assert should_poll_status(False, False) is False


class TestShouldDeferStatus:
    """Lines 303-318: returns bool(status_is_cuda) and bool(defer_on)."""

    def test_both_true(self):
        assert should_defer_status(True, True) is True

    def test_cuda_false(self):
        assert should_defer_status(False, True) is False

    def test_defer_off(self):
        assert should_defer_status(True, False) is False

    def test_both_false(self):
        assert should_defer_status(False, False) is False


# ---------------------------------------------------------------------------
# pause_polling context manager -- capture depth (lines 222-253)
# ---------------------------------------------------------------------------


class TestPausePolling:
    """Test the _PausePolling reentrant context manager without a GPU."""

    def test_basic_enter_exit_toggles_paused(self, monkeypatch):
        """polling_paused flips True during __enter__ and back to False."""
        # We need to reset any pre-existing depth; _PausePolling uses a global.
        monkeypatch.setattr(
            "sglang.srt.distributed.device_communicators.barlink_abort_gate._capture_depth",
            0,
        )
        from sglang.srt.distributed.device_communicators import barlink_abort_gate as m

        ctx = pause_polling()
        assert m.polling_paused() is False  # before
        with ctx:
            assert m.polling_paused() is True  # inside
        assert m.polling_paused() is False  # after

    def test_reentrant_nested(self, monkeypatch):
        """Two pause_polling() contexts nested -> depth 2, unpauses fully only after both exit."""
        from sglang.srt.distributed.device_communicators import barlink_abort_gate as m

        monkeypatch.setattr(m, "_capture_depth", 0)

        with pause_polling():
            assert m.polling_paused() is True
            with pause_polling():
                assert m.polling_paused() is True
            # inner exited but outer still holds
            assert m.polling_paused() is True
        # both exited
        assert m.polling_paused() is False

    def test_pause_polling_does_not_swallow_exceptions(self, monkeypatch):
        """__exit__ returns None, so exceptions propagate (line 235)."""
        from sglang.srt.distributed.device_communicators import barlink_abort_gate as m

        monkeypatch.setattr(m, "_capture_depth", 0)

        with pytest.raises(ValueError):
            with pause_polling():
                raise ValueError("must propagate")
        # depth still 0 after exception
        assert m.polling_paused() is False


# ---------------------------------------------------------------------------
# Registry CRUD -- register, unregister, registered, reset_for_test
# ---------------------------------------------------------------------------


class TestRegistry:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        reset_for_test()
        yield
        reset_for_test()

    def test_register_then_listed(self):
        obj = object()
        register(obj)
        assert registered() == [obj]

    def test_unregister_removes(self):
        obj = object()
        register(obj)
        unregister(obj)
        assert registered() == []

    def test_duplicate_register_noop(self):
        obj = object()
        register(obj)
        register(obj)
        assert registered() == [obj]

    def test_unregister_missing_noop(self):
        obj = object()
        unregister(obj)  # should not raise
        assert registered() == []

    def test_registered_returns_copy(self):
        obj = object()
        register(obj)
        snap = registered()
        snap.append("poison")
        assert "poison" not in registered()

    def test_reset_for_test_clears(self):
        register(object())
        register(object())
        reset_for_test()
        assert registered() == []

    def test_multiple_transports(self):
        a, b, c = object(), object(), object()
        register(a)
        register(b)
        register(c)
        assert len(registered()) == 3
        unregister(b)
        assert set(registered()) == {a, c}


# ---------------------------------------------------------------------------
# poll_status_words -- returns 0 when registry is empty
# ---------------------------------------------------------------------------


class TestPollStatusWords:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        reset_for_test()
        monkeypatch.setenv(ENV_ENABLE, "1")
        yield
        reset_for_test()

    def test_empty_registry_returns_zero(self):
        assert poll_status_words() == 0

    def test_disabled_does_not_poll(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "0")
        assert poll_status_words() == 0

    def test_paused_does_not_poll(self, monkeypatch):
        from sglang.srt.distributed.device_communicators import barlink_abort_gate as m

        m._capture_depth = 1
        monkeypatch.setenv(ENV_ENABLE, "1")
        try:
            assert poll_status_words() == 0
        finally:
            m._capture_depth = 0

    def test_transport_without_poll_method_skipped(self):
        """Transport missing poll_status_word is skipped (line 276-278)."""
        obj = object()  # no poll_status_word attr
        register(obj)
        # Should not raise, just return 0
        result = poll_status_words()
        assert result == 0

    def test_transport_with_poll_method_summed(self, monkeypatch):
        """When a transport has poll_status_word returning True, it's counted."""
        obj = type("FakeTransport", (), {})()
        obj.poll_status_word = lambda: True
        register(obj)
        assert poll_status_words() == 1

    def test_transport_poll_exception_handled(self, monkeypatch, caplog):
        """poll_status_word that raises -> logged, not re-raised (lines 282-283)."""
        obj = type("FakeTransport", (), {})()
        obj.poll_status_word = lambda: 1 / 0  # ZeroDivisionError
        register(obj)
        result = poll_status_words()
        # Should return 0 (the exception transport is skipped), not crash
        assert result == 0
        assert "barlink-BAR1 status poll failed" in caplog.text


# ---------------------------------------------------------------------------
# check_aborts and check_after_graph_replay -- empty path
# ---------------------------------------------------------------------------


class TestCheckAborts:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        reset_for_test()
        monkeypatch.setenv(ENV_ENABLE, "1")
        yield
        reset_for_test()

    def test_empty_registry_no_op(self):
        """No transport registered -> immediate return (line 359-360)."""
        check_aborts("test")  # must not raise

    def test_disabled_no_op(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "0")
        check_aborts("test")  # must not raise

    def test_transport_with_check_aborted_called(self):
        """Transport with check_aborted method -> it gets called (lines 363-366)."""
        called = []
        obj = type("FakeTransport", (), {})()
        obj.check_aborted = lambda where: called.append(where)
        register(obj)
        check_aborts("my-label")
        assert called == ["my-label"]

    def test_transport_without_check_aborted_skipped(self):
        """Transport missing check_aborted is skipped (line 364-365)."""
        register(object())
        check_aborts("test")  # must not raise


class TestCheckAfterGraphReplay:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        reset_for_test()
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_REPLAY, "1")
        yield
        reset_for_test()

    def test_empty_registry_no_op(self):
        check_after_graph_replay()  # must not raise

    def test_replay_disabled_skips(self, monkeypatch):
        monkeypatch.setenv(ENV_REPLAY, "0")
        register(object())  # register something so we pass the empty check
        check_after_graph_replay()  # must not raise (line 380-381)

    def test_main_enable_disabled_skips(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "0")
        register(object())
        check_after_graph_replay()  # must not raise (replay_check_enabled ANDs enable)


# ---------------------------------------------------------------------------
# replay_check_enabled -- AND logic
# ---------------------------------------------------------------------------


class TestReplayCheckEnabled:
    """replay_check_enabled = abort_check_enabled() AND _env_flag(ENV_REPLAY, True)"""

    def test_both_true(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_REPLAY, "1")
        assert replay_check_enabled() is True

    def test_enable_false_replay_true(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "0")
        monkeypatch.setenv(ENV_REPLAY, "1")
        assert replay_check_enabled() is False

    def test_enable_true_replay_false(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_REPLAY, "0")
        assert replay_check_enabled() is False

    def test_both_false(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "0")
        monkeypatch.setenv(ENV_REPLAY, "0")
        assert replay_check_enabled() is False
