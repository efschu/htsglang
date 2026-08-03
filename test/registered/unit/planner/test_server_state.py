# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for server_state.py -- the four-state server diagnosis.

Hermetic: no server, no socket, no GPU. Every probe is injected.

The FALSIFIER these tests exist for is ``test_no_metrics_is_never_claimed_
without_an_api_probe``: the defect being fixed was a page that rendered
"started without --enable-metrics" while the scrape had returned
"connection refused" -- two different states of the world collapsed into one
guessed cause. Delete the ``api.ok`` guard in ``classify`` and that test goes
red.
"""

import itertools
import unittest
import urllib.error

from sglang.srt.planner import server_state as ss
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _p(ok, path="/x", **kw):
    return ss.Probe(ok=ok, path=path, **kw)


class _FakeResp:
    def __init__(self, code):
        self._code = code

    def getcode(self):
        return self._code

    def read(self):
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener(map_by_path, calls=None):
    """urlopen stand-in: path suffix -> HTTP code, an Exception, or a callable."""

    def _open(url, timeout=None):
        if calls is not None:
            calls.append(url)
        for suffix, out in map_by_path.items():
            if url.endswith(suffix):
                if isinstance(out, Exception):
                    raise out
                return _FakeResp(out)
        raise ConnectionRefusedError("Connection refused")

    return _open


# ---------------------------------------------------------------------------
# The state machine, pure.
# ---------------------------------------------------------------------------
class TestClassify(CustomTestCase):
    def test_all_four_states_are_reachable(self):
        got = {
            ss.classify(_p(True), _p(True), ss.BootEvidence()),
            ss.classify(_p(True), _p(False), ss.BootEvidence()),
            ss.classify(_p(False), _p(False),
                        ss.BootEvidence(starting=True, source="managed")),
            ss.classify(_p(False), _p(False), ss.BootEvidence()),
        }
        self.assertEqual(got, set(ss.STATES))

    def test_no_metrics_is_never_claimed_without_an_api_probe(self):
        """THE falsifier. Over the exhaustive product of inputs, the state that
        blames --enable-metrics is unreachable while the API probe failed."""
        for api_ok, met_ok, booting in itertools.product(
                (True, False), (True, False), (True, False)):
            state = ss.classify(
                _p(api_ok), _p(met_ok),
                ss.BootEvidence(starting=booting, source="managed" if booting
                                else "none"))
            if state == ss.RUNNING_NO_METRICS:
                self.assertTrue(
                    api_ok,
                    "RUNNING_NO_METRICS reached with a FAILED api probe "
                    f"(api_ok={api_ok} met_ok={met_ok} booting={booting})")

    def test_refused_scrape_alone_is_not_a_flag_diagnosis(self):
        """A connection-refused scrape and a 404 scrape differ ONLY in what the
        API probe says -- which is exactly what the old code could not see."""
        refused = _p(False, "/metrics", error="ConnectionRefusedError: refused")
        four04 = _p(False, "/metrics", status=404, error="HTTP 404")
        dead = ss.build(_p(False, "/get_model_info",
                           error="ConnectionRefusedError: refused"), refused)
        alive = ss.build(_p(True, "/get_model_info", status=200), four04)
        self.assertEqual(dead.state, ss.NOT_RUNNING)
        self.assertEqual(alive.state, ss.RUNNING_NO_METRICS)
        self.assertNotIn("enable-metrics", dead.headline + dead.detail)
        self.assertIn("--enable-metrics", alive.headline)

    def test_starting_outranks_not_running_but_never_a_live_state(self):
        boot = ss.BootEvidence(starting=True, source="port-open")
        self.assertEqual(
            ss.classify(_p(False), _p(False), boot), ss.STARTING)
        # Evidence of a boot must not override an answering API.
        self.assertEqual(ss.classify(_p(True), _p(False), boot),
                         ss.RUNNING_NO_METRICS)
        self.assertEqual(ss.classify(_p(True), _p(True), boot),
                         ss.RUNNING_WITH_METRICS)

    def test_running_property_covers_exactly_the_two_live_states(self):
        for state, api, met in (
                (ss.NOT_RUNNING, False, False),
                (ss.RUNNING_NO_METRICS, True, False),
                (ss.RUNNING_WITH_METRICS, True, True)):
            built = ss.build(_p(api), _p(met))
            self.assertEqual(built.state, state)
            self.assertEqual(built.running, state.startswith("running"))

    def test_not_running_says_nothing_about_launch_flags(self):
        dead = ss.build(_p(False), _p(False))
        text = (dead.headline + " " + dead.detail).lower()
        self.assertIn("no server running", dead.headline.lower())
        self.assertNotIn("enable-metrics", text)


# ---------------------------------------------------------------------------
# Boot evidence.
# ---------------------------------------------------------------------------
class TestBootEvidence(CustomTestCase):
    def test_managed_boot_is_known_without_probing(self):
        seen = []
        ev = ss.boot_evidence(
            managed_state="booting", host="h", port=1,
            tcp_probe=lambda h, p: seen.append((h, p)) or True)
        self.assertTrue(ev.starting)
        self.assertEqual(ev.source, "managed")
        self.assertEqual(seen, [], "managed state must short-circuit the probe")

    def test_managed_ready_is_not_a_boot(self):
        ev = ss.boot_evidence(managed_state="ready")
        self.assertFalse(ev.starting)

    def test_foreign_boot_needs_the_port_to_accept(self):
        yes = ss.boot_evidence(host="h", port=30000, tcp_probe=lambda h, p: True)
        no = ss.boot_evidence(host="h", port=30000, tcp_probe=lambda h, p: False)
        self.assertEqual((yes.starting, yes.source), (True, "port-open"))
        self.assertEqual((no.starting, no.source), (False, "none"))
        self.assertIn("30000", yes.detail)

    def test_no_evidence_means_no_starting_claim(self):
        ev = ss.boot_evidence()
        self.assertFalse(ev.starting)
        self.assertEqual(ev.source, "none")


# ---------------------------------------------------------------------------
# Probes: bounded, never raising, status-preserving.
# ---------------------------------------------------------------------------
class TestProbes(CustomTestCase):
    def test_http_error_keeps_its_status(self):
        err = urllib.error.HTTPError("u", 404, "nf", None, None)
        pr = ss.probe_http("http://h:1", "/metrics", opener=_opener({"/metrics": err}))
        self.assertFalse(pr.ok)
        self.assertEqual(pr.status, 404)

    def test_transport_failure_is_a_result_not_an_exception(self):
        pr = ss.probe_http("http://h:1", "/metrics", opener=_opener({}))
        self.assertFalse(pr.ok)
        self.assertIsNone(pr.status)
        self.assertIn("ConnectionRefused", pr.error)

    def test_api_probe_falls_through_to_health(self):
        calls = []
        pr = ss.probe_api("http://h:1", opener=_opener(
            {"/get_model_info": ConnectionRefusedError("x"), "/health": 200},
            calls))
        self.assertTrue(pr.ok)
        self.assertEqual(pr.path, "/health")
        self.assertEqual(len(calls), 2)

    def test_api_probe_stops_at_the_first_answer(self):
        calls = []
        pr = ss.probe_api("http://h:1",
                          opener=_opener({"/get_model_info": 200}, calls))
        self.assertTrue(pr.ok)
        self.assertEqual(len(calls), 1, "no probe is spent after a 200")


# ---------------------------------------------------------------------------
# resolve(): the whole tick, incl. the cost contract.
# ---------------------------------------------------------------------------
class TestResolve(CustomTestCase):
    def test_healthy_path_spends_no_extra_probe(self):
        calls = []
        st = ss.resolve("http://h:1", opener=_opener({"/metrics": 200}, calls))
        self.assertEqual(st.state, ss.RUNNING_WITH_METRICS)
        self.assertFalse(st.api.attempted)
        self.assertIn("same HTTP server", st.api.reason)
        self.assertEqual(len(calls), 1)

    def test_metrics_404_plus_live_api_is_state_three(self):
        st = ss.resolve("http://h:1", opener=_opener({
            "/metrics": urllib.error.HTTPError("u", 404, "nf", None, None),
            "/get_model_info": 200}))
        self.assertEqual(st.state, ss.RUNNING_NO_METRICS)
        self.assertTrue(st.api.attempted)
        self.assertTrue(st.api.ok)

    def test_everything_refused_is_not_running(self):
        st = ss.resolve("http://h:1", opener=_opener({}),
                        tcp_probe=lambda h, p: False, host="h", port=1)
        self.assertEqual(st.state, ss.NOT_RUNNING)

    def test_managed_boot_shows_starting_from_the_first_tick(self):
        st = ss.resolve("http://h:1", opener=_opener({}),
                        managed_state="booting", managed_detail="pid 7.")
        self.assertEqual(st.state, ss.STARTING)
        self.assertEqual(st.boot.source, "managed")
        self.assertIn("starting", st.headline.lower())

    def test_starting_to_running_transition(self):
        """Same target, two consecutive ticks: booting -> ready with metrics."""
        boot = ss.resolve("http://h:1", opener=_opener({}),
                          managed_state="booting")
        up = ss.resolve("http://h:1", opener=_opener({"/metrics": 200}),
                        managed_state="ready")
        self.assertEqual(boot.state, ss.STARTING)
        self.assertEqual(up.state, ss.RUNNING_WITH_METRICS)

    def test_starting_to_running_no_metrics_transition(self):
        boot = ss.resolve("http://h:1", opener=_opener({}),
                          host="h", port=1, tcp_probe=lambda h, p: True)
        up = ss.resolve("http://h:1", opener=_opener({"/get_model_info": 200}),
                        host="h", port=1, tcp_probe=lambda h, p: True)
        self.assertEqual(boot.state, ss.STARTING)
        self.assertEqual(up.state, ss.RUNNING_NO_METRICS)

    def test_no_endpoint_spends_no_probe_at_all(self):
        st = ss.resolve(None)
        self.assertEqual(st.state, ss.NOT_RUNNING)
        self.assertFalse(st.api.attempted)
        self.assertFalse(st.metrics.attempted)

    def test_injected_metrics_probe_is_not_rescraped(self):
        calls = []
        st = ss.resolve("http://h:1",
                        metrics=_p(False, "/metrics", error="refused"),
                        opener=_opener({"/get_model_info": 200}, calls))
        self.assertEqual(st.state, ss.RUNNING_NO_METRICS)
        self.assertEqual([c for c in calls if c.endswith("/metrics")], [])

    def test_json_shape_is_complete(self):
        j = ss.resolve("http://h:1", opener=_opener({"/metrics": 200})).to_json()
        self.assertEqual(
            set(j),
            {"state", "running", "headline", "detail", "api", "metrics", "boot"})
        self.assertIn(j["state"], ss.STATES)


if __name__ == "__main__":
    unittest.main()
