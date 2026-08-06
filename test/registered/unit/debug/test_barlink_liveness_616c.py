# SPDX-License-Identifier: Apache-2.0
"""Hermetic unit tests for barlink_liveness -- no CUDA, no distributed runtime.

Tests cover the pure-python surface: env-var parsing, pid-liveness probes,
peer-identity formatting, the registry CRUD, bounded_poll success/timeout/
peer-lost paths, error-message formatting, and describe_config.

Untestable hermetically (require torch / real distributed group / GPU):
  - PeerTable.exchange()  (needs cpu_group + dist.all_gather_object)
  - AbortWindow._build()  (needs torch + CUDA runtime ctypes)
  - bounded_barrier()     (needs dist group)
  - bounded_collective()  (needs dist group)
  - bounded_device_sync() (needs CUDA)
  - install()             (needs cpu_group)
  - PeerWatchdog._run()   (imports barlink_abort_gate, starts a daemon thread)
"""

from __future__ import annotations

import os
import socket

import pytest

from sglang.srt.distributed.device_communicators.barlink_liveness import (
    ALIVE,
    DEAD,
    ENV_ENABLE,
    ENV_PROBE_S,
    ENV_TIMEOUT_S,
    ENV_WATCHDOG,
    UNKNOWN,
    CollectiveTimeoutError,
    PeerIdentity,
    PeerLivenessError,
    PeerLostError,
    PeerTable,
    PeerWatchdog,
    _dead_peer_error,
    _env_flag,
    _env_float,
    _timeout_error,
    any_dead_peers,
    bounded_poll,
    describe_config,
    local_identity,
    liveness_enabled,
    pid_alive,
    probe_interval_s,
    register_peer_table,
    registered_tables,
    reset_for_test,
    trip_all_abort_windows,
    wait_timeout_s,
    watchdog_enabled,
)

_H = socket.gethostname()  # real hostname; PeerTable.state() compares against it


# ---------------------------------------------------------------------------
# conftest-style cleanup: reset the module-level registry between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_for_test()
    yield
    reset_for_test()


# ---------------------------------------------------------------------------
# _env_flag -- pure string classification
# ---------------------------------------------------------------------------


class TestEnvFlag:
    """_env_flag(name, default) returns default when unset, True when not in _FALSE."""

    def test_unset_returns_default_true(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_ENV_XYZ", raising=False)
        assert _env_flag("NONEXISTENT_ENV_XYZ", True) is True

    def test_unset_returns_default_false(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_ENV_XYZ", raising=False)
        assert _env_flag("NONEXISTENT_ENV_XYZ", False) is False

    def test_falsy_values(self, monkeypatch):
        for val in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("TEST_FLAG", val)
            assert _env_flag("TEST_FLAG", True) is False, f"expected False for {val!r}"

    def test_truthy_value(self, monkeypatch):
        monkeypatch.setenv("TEST_FLAG", "1")
        assert _env_flag("TEST_FLAG", False) is True

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("TEST_FLAG", "  FALSE  ")
        assert _env_flag("TEST_FLAG", True) is False

    def test_random_string_is_truthy(self, monkeypatch):
        """Anything not in _FALSE is truthy (line 132)."""
        monkeypatch.setenv("TEST_FLAG", "xyzzy")
        assert _env_flag("TEST_FLAG", False) is True


# ---------------------------------------------------------------------------
# _env_float -- numeric env parsing with fallback
# ---------------------------------------------------------------------------


class TestEnvFloat:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_FLOAT_XYZ", raising=False)
        assert _env_float("NONEXISTENT_FLOAT_XYZ", 3.14) == 3.14

    def test_valid_number(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "2.5")
        assert _env_float("TEST_FLOAT", 0.0) == 2.5

    def test_invalid_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("TEST_FLOAT", "not-a-number")
        result = _env_float("TEST_FLOAT", 9.9)
        assert result == 9.9
        assert "is not a number" in caplog.text


# ---------------------------------------------------------------------------
# liveness_enabled / watchdog_enabled -- public env wrappers
# ---------------------------------------------------------------------------


class TestLivenessEnabled:
    def test_default_true(self, monkeypatch):
        monkeypatch.delenv(ENV_ENABLE, raising=False)
        assert liveness_enabled() is True

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "0")
        assert liveness_enabled() is False


class TestWatchdogEnabled:
    """watchdog_enabled = liveness_enabled AND _env_flag(ENV_WATCHDOG, True)."""

    def test_both_on(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.delenv(ENV_WATCHDOG, raising=False)
        assert watchdog_enabled() is True

    def test_liveness_off_implies_watchdog_off(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "0")
        monkeypatch.setenv(ENV_WATCHDOG, "1")
        assert watchdog_enabled() is False

    def test_watchdog_explicit_off(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_WATCHDOG, "0")
        assert watchdog_enabled() is False


# ---------------------------------------------------------------------------
# probe_interval_s / wait_timeout_s
# ---------------------------------------------------------------------------


class TestProbeIntervalS:
    def test_default(self, monkeypatch):
        monkeypatch.delenv(ENV_PROBE_S, raising=False)
        assert probe_interval_s() == 1.0

    def test_explicit(self, monkeypatch):
        monkeypatch.setenv(ENV_PROBE_S, "0.5")
        assert probe_interval_s() == 0.5

    def test_negative_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv(ENV_PROBE_S, "-5")
        assert probe_interval_s() == 0.0


class TestWaitTimeoutS:
    def test_default_120(self, monkeypatch):
        monkeypatch.delenv(ENV_TIMEOUT_S, raising=False)
        assert wait_timeout_s() == 120.0

    def test_explicit(self, monkeypatch):
        monkeypatch.setenv(ENV_TIMEOUT_S, "300")
        assert wait_timeout_s() == 300.0

    def test_zero_or_negative_returns_zero(self, monkeypatch):
        monkeypatch.setenv(ENV_TIMEOUT_S, "0")
        assert wait_timeout_s() == 0.0


# ---------------------------------------------------------------------------
# pid_alive -- os.kill(pid, 0) wrapper
# ---------------------------------------------------------------------------


class TestPidAlive:
    def test_own_pid_is_alive(self):
        assert pid_alive(os.getpid()) is True

    def test_impossible_pid_is_dead(self):
        """A very large pid that can never exist."""
        assert pid_alive(999999999) is False

    def test_pid_zero_is_dead(self):
        assert pid_alive(0) is False

    def test_negative_pid_is_dead(self):
        assert pid_alive(-1) is False


# ---------------------------------------------------------------------------
# _boot_marker -- reads /proc/<pid>/stat
# ---------------------------------------------------------------------------


class TestBootMarker:
    def test_own_pid_returns_non_empty(self):
        """On Linux /proc is available, so our own pid yields a marker."""
        from sglang.srt.distributed.device_communicators.barlink_liveness import (
            _boot_marker,
        )

        marker = _boot_marker(os.getpid())
        assert isinstance(marker, str)
        assert len(marker) > 0

    def test_impossible_pid_returns_empty(self):
        from sglang.srt.distributed.device_communicators.barlink_liveness import (
            _boot_marker,
        )

        marker = _boot_marker(999999999)
        assert marker == ""


# ---------------------------------------------------------------------------
# PeerIdentity.describe() -- string formatting
# ---------------------------------------------------------------------------


class TestPeerIdentityDescribe:
    def test_describe_format(self):
        p = PeerIdentity(rank=2, host="node1", pid=1234, boot="abc")
        assert p.describe() == "rank 2 (node1, pid 1234)"

    def test_describe_with_spaces_in_host(self):
        p = PeerIdentity(rank=0, host="my host", pid=1, boot="")
        assert p.describe() == "rank 0 (my host, pid 1)"


# ---------------------------------------------------------------------------
# local_identity(rank) -- builds PeerIdentity from current process
# ---------------------------------------------------------------------------


class TestLocalIdentity:
    def test_returns_expected_fields(self):
        ident = local_identity(rank=5)
        assert ident.rank == 5
        assert len(ident.host) > 0
        assert ident.pid == os.getpid()
        assert isinstance(ident.boot, str)


# ---------------------------------------------------------------------------
# PeerTable -- state queries, dead_peers, describe
# ---------------------------------------------------------------------------


class TestPeerTable:
    """All test entries use the real hostname (_H) so that
    PeerTable.state() treats them as same-host peers (line 313).
    Boot markers are left empty ('') to skip the /proc boot check
    (line 317-319) and let the pid_alive gate decide alone."""

    @pytest.fixture
    def table(self):
        alive = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        dead = PeerIdentity(rank=1, host=_H, pid=999999999, boot="")
        other_host = PeerIdentity(rank=2, host="remote-host", pid=1111, boot="")
        opaque = PeerIdentity(rank=3, host="", pid=0, boot="")
        return PeerTable([alive, dead, other_host, opaque], self_rank=0)

    def test_self_is_alive(self, table):
        assert table.state(table.entries[0]) == ALIVE

    def test_dead_pid_on_same_host(self, table):
        assert table.state(table.entries[1]) == DEAD

    def test_other_host_is_unknown(self, table):
        assert table.state(table.entries[2]) == UNKNOWN

    def test_opaque_entry_is_unknown(self, table):
        assert table.state(table.entries[3]) == UNKNOWN

    def test_dead_peers_returns_only_dead(self, table):
        dead = table.dead_peers()
        assert len(dead) == 1
        assert dead[0].rank == 1

    def test_describe_dead_text(self, table):
        assert "rank 1" in table.describe_dead()

    def test_describe_includes_all_states(self, table):
        desc = table.describe()
        assert ALIVE in desc
        assert DEAD in desc
        assert UNKNOWN in desc

    def test_no_dead_peers_empty_string(self):
        alive = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        pt = PeerTable([alive], self_rank=0)
        assert pt.describe_dead() == ""


# ---------------------------------------------------------------------------
# Registry -- register_peer_table, registered_tables, reset_for_test
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_list(self):
        p = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        t = PeerTable([p], self_rank=0)
        register_peer_table(t)
        tables = registered_tables()
        assert len(tables) == 1
        assert tables[0] is t

    def test_duplicate_register_is_noop(self):
        p = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        t = PeerTable([p], self_rank=0)
        register_peer_table(t)
        register_peer_table(t)
        assert len(registered_tables()) == 1

    def test_any_dead_peers_across_tables(self):
        alive = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        dead = PeerIdentity(rank=1, host=_H, pid=999999999, boot="")
        t1 = PeerTable([alive, dead], self_rank=0)
        register_peer_table(t1)
        dead_list = any_dead_peers()
        assert len(dead_list) == 1
        assert dead_list[0].rank == 1

    def test_any_dead_peers_deduplicates(self):
        """Same (host, pid, rank) appearing in two tables counts once."""
        dead = PeerIdentity(rank=1, host=_H, pid=999999999, boot="")
        alive = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        t1 = PeerTable([alive, dead], self_rank=0)
        t2 = PeerTable([alive, dead], self_rank=0)
        register_peer_table(t1)
        register_peer_table(t2)
        dead_list = any_dead_peers()
        assert len(dead_list) == 1  # deduplicated


# ---------------------------------------------------------------------------
# Error message formatting -- _dead_peer_error, _timeout_error
# ---------------------------------------------------------------------------


class TestErrorMessageFormatting:
    def test_dead_peer_error_contents(self):
        dead = PeerIdentity(rank=3, host="srv1", pid=7777, boot="")
        msg = _dead_peer_error("all-reduce", [dead], 42.5)
        assert "all-reduce" in msg
        assert "rank 3" in msg
        assert "42.5" in msg
        assert ENV_ENABLE in msg

    def test_dead_peer_error_multiple_ranks(self):
        d1 = PeerIdentity(rank=1, host="h", pid=10, boot="")
        d2 = PeerIdentity(rank=2, host="h", pid=20, boot="")
        msg = _dead_peer_error("gather", [d1, d2], 0.0)
        assert "rank 1" in msg
        assert "rank 2" in msg

    def test_timeout_error_with_table(self):
        alive = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        t = PeerTable([alive], self_rank=0)
        msg = _timeout_error("bcast", 120.0, t)
        assert "bcast" in msg
        assert "120" in msg
        assert "Peer census" in msg

    def test_timeout_error_without_table(self):
        msg = _timeout_error("recv", 60.0, None)
        assert "recv" in msg
        assert "Peer census" not in msg


# ---------------------------------------------------------------------------
# bounded_poll -- success, timeout, peer-lost, disabled
# ---------------------------------------------------------------------------


class TestBoundedPoll:
    def test_immediate_success(self, monkeypatch):
        """Predicate is True at entry: returns before any clock read (line 748-749)."""
        monkeypatch.setenv(ENV_ENABLE, "1")
        counter = [0]

        def ready():
            counter[0] += 1
            return True

        bounded_poll(ready, "fast")
        assert counter[0] == 1

    def test_disabled_blocks_until_ready(self, monkeypatch):
        """With liveness disabled, it falls back to plain while-loop (lines 750-754)."""
        monkeypatch.setenv(ENV_ENABLE, "0")
        counter = [0]

        def ready():
            counter[0] += 1
            if counter[0] >= 5:
                return True
            return False

        bounded_poll(ready, "disabled", sleep=False)
        assert counter[0] == 5

    def test_timeout_raises_CollectiveTimeoutError(self, monkeypatch):
        """When deadline expires and no peer is dead -> CollectiveTimeoutError."""
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_PROBE_S, "0")

        def never():
            return False

        with pytest.raises(CollectiveTimeoutError) as exc:
            bounded_poll(
                never,
                "slow-op",
                timeout_s=0.01,
                table=None,
                sleep=False,
            )
        assert "slow-op" in str(exc.value)
        assert "no peer could be proven dead" in str(exc.value)

    def test_peer_lost_raises_PeerLostError(self, monkeypatch):
        """A PeerTable reporting DEAD peers triggers PeerLostError."""
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_PROBE_S, "0")

        dead = PeerIdentity(rank=1, host=_H, pid=999999999, boot="")
        alive = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        t = PeerTable([alive, dead], self_rank=0)

        def never():
            return False

        with pytest.raises(PeerLostError) as exc:
            bounded_poll(
                never,
                "all-gather",
                timeout_s=5.0,
                table=t,
                sleep=False,
            )
        assert "all-gather" in str(exc.value)
        assert "rank 1" in str(exc.value)
        assert "no longer exists" in str(exc.value)

    def test_on_abort_called_on_timeout(self, monkeypatch):
        """on_abort callback fires before the exception is raised."""
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_PROBE_S, "0")

        abort_log = []

        def on_abort():
            abort_log.append(True)

        def never():
            return False

        with pytest.raises(CollectiveTimeoutError):
            bounded_poll(
                never,
                "sync",
                timeout_s=0.01,
                table=None,
                on_abort=on_abort,
                sleep=False,
            )
        assert abort_log == [True]

    def test_error_hierarchy(self):
        """PeerLostError and CollectiveTimeoutError inherit from PeerLivenessError."""
        assert issubclass(PeerLostError, PeerLivenessError)
        assert issubclass(CollectiveTimeoutError, PeerLivenessError)
        assert issubclass(PeerLivenessError, RuntimeError)


# ---------------------------------------------------------------------------
# describe_config -- public settings summary
# ---------------------------------------------------------------------------


class TestDescribeConfig:
    def test_contains_env_keys(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "1")
        monkeypatch.setenv(ENV_TIMEOUT_S, "120")
        monkeypatch.setenv(ENV_PROBE_S, "1")
        monkeypatch.delenv(ENV_WATCHDOG, raising=False)
        cfg = describe_config()
        assert ENV_ENABLE in cfg
        assert ENV_TIMEOUT_S in cfg
        assert ENV_PROBE_S in cfg
        assert ENV_WATCHDOG in cfg

    def test_disabled_config(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "0")
        cfg = describe_config()
        assert "0" in cfg


# ---------------------------------------------------------------------------
# trip_all_abort_windows -- degraded path (no CUDA = windows never registered,
# but the function itself must not crash on an empty list)
# ---------------------------------------------------------------------------


class TestTripAllWindows:
    def test_empty_registry_no_crash(self):
        """When no windows are registered, the function is a no-op."""
        trip_all_abort_windows("test-reason")  # must not raise


# ---------------------------------------------------------------------------
# PeerWatchdog.probe_once -- pure host-side probe (no thread, no CUDA)
# ---------------------------------------------------------------------------


class TestPeerWatchdogProbeOnce:
    def test_no_dead_returns_empty(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "1")
        wd = PeerWatchdog(interval_s=999)
        dead = wd.probe_once()
        assert dead == []
        assert wd.trips == 0

    def test_dead_peer_trips_and_counts(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLE, "1")
        alive = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        dead = PeerIdentity(rank=1, host=_H, pid=999999999, boot="")
        t = PeerTable([alive, dead], self_rank=0)
        register_peer_table(t)

        wd = PeerWatchdog(interval_s=999)
        dead_list = wd.probe_once()
        assert len(dead_list) == 1
        assert dead_list[0].rank == 1
        assert wd.trips == 1

    def test_probe_returns_PeerIdentity_list(self, monkeypatch):
        """probe_once returns a list of PeerIdentity objects."""
        monkeypatch.setenv(ENV_ENABLE, "1")
        alive = PeerIdentity(rank=0, host=_H, pid=os.getpid(), boot="")
        dead = PeerIdentity(rank=3, host=_H, pid=999999999, boot="")
        t = PeerTable([alive, dead], self_rank=0)
        register_peer_table(t)

        wd = PeerWatchdog(interval_s=999)
        dead = wd.probe_once()
        assert all(isinstance(e, PeerIdentity) for e in dead)
