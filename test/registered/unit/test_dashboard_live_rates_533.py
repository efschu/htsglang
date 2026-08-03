"""#533: the dashboard's live-rate baselines must survive concurrent monitors.

Reported symptom: the dashboard tiles were "constantly empty" while the served
model was under real load. Measured root: ONE module-global delta baseline plus
a single target key, wiped on every key change. The landing page resolves its
target three ways (explicit ?endpoint=, supervisor-managed, auto-detected), so
two clients resolving differently alternate the key and reset each other's
baseline on every request -- and ``live_metrics._rates`` returns None without a
baseline. Before the fix, 8/8 consecutive polls at a stable key returned
``rates: null`` while ``generation_tokens_total`` climbed 24253 -> 27134.

Hermetic: no server, no CUDA, no NVML.
"""

import unittest

from sglang.srt.planner import webui


class _BaselineTestBase(unittest.TestCase):
    def setUp(self):
        webui._LANDING_STATE_BY_KEY.clear()
        self.addCleanup(webui._LANDING_STATE_BY_KEY.clear)

    @staticmethod
    def state(t, gen):
        return {"t": t, "counters": {"generation_tokens_total": gen}}


class TestConcurrentMonitorsAreIndependent(_BaselineTestBase):
    def test_two_keys_do_not_reset_each_other(self):
        """The exact production shape: a browser on one key and a second
        poller on another, interleaved. Both must keep their own baseline."""
        a = ("detected", "http://127.0.0.1:30030")
        b = ("explicit", "http://localhost:30030")
        webui._store_landing_baseline(a, None, self.state(100.0, 1000), None)
        webui._store_landing_baseline(b, None, self.state(100.1, 1000), None)
        # A polls again 2 s later; its baseline must still be A's, not wiped
        # by B's intervening poll.
        self.assertEqual(webui._LANDING_STATE_BY_KEY[a]["t"], 100.0)
        self.assertEqual(webui._LANDING_STATE_BY_KEY[b]["t"], 100.1)
        webui._store_landing_baseline(a, webui._LANDING_STATE_BY_KEY[a],
                                      self.state(102.0, 1160), 2.0)
        self.assertEqual(webui._LANDING_STATE_BY_KEY[a]["t"], 102.0)
        self.assertEqual(webui._LANDING_STATE_BY_KEY[b]["t"], 100.1)

    def test_a_third_target_evicts_oldest_not_newest(self):
        for i in range(webui._LANDING_STATE_MAX_KEYS + 2):
            webui._store_landing_baseline(("detected", f"h{i}"), None,
                                          self.state(float(i), i), None)
        self.assertLessEqual(len(webui._LANDING_STATE_BY_KEY),
                             webui._LANDING_STATE_MAX_KEYS)
        # The most recent key survives; the first one is gone.
        self.assertIn(("detected", "h9"), webui._LANDING_STATE_BY_KEY)
        self.assertNotIn(("detected", "h0"), webui._LANDING_STATE_BY_KEY)


class TestShortWindowDoesNotCollapseTheBaseline(_BaselineTestBase):
    def test_sub_floor_scrape_keeps_the_older_baseline(self):
        """A second poller landing 0.2 s after the first must not shrink the
        window to something that reads as 0 tok/s."""
        key = ("detected", "http://127.0.0.1:30030")
        webui._store_landing_baseline(key, None, self.state(100.0, 1000), None)
        stored = webui._store_landing_baseline(
            key, webui._LANDING_STATE_BY_KEY[key], self.state(100.2, 1000), 0.2)
        self.assertFalse(stored)
        self.assertEqual(webui._LANDING_STATE_BY_KEY[key]["t"], 100.0)

    def test_window_beyond_the_floor_advances(self):
        key = ("detected", "http://127.0.0.1:30030")
        webui._store_landing_baseline(key, None, self.state(100.0, 1000), None)
        stored = webui._store_landing_baseline(
            key, webui._LANDING_STATE_BY_KEY[key], self.state(101.0, 1080), 1.0)
        self.assertTrue(stored)
        self.assertEqual(webui._LANDING_STATE_BY_KEY[key]["t"], 101.0)

    def test_first_sample_is_always_stored(self):
        """No baseline yet -> store regardless of dt, or rates never start."""
        key = ("detected", "x")
        self.assertTrue(webui._store_landing_baseline(key, None,
                                                      self.state(1.0, 5), 0.01))


class TestRatesActuallyComputeFromStoredBaseline(_BaselineTestBase):
    """End-to-end on the delta math: a stored baseline must yield the rate a
    hand calculation predicts, so the storage fix is connected to the number
    the tile shows."""

    def test_decode_rate_matches_hand_calculation(self):
        from sglang.srt.planner import live_metrics

        prev = {"generation_tokens_total": 1000.0, "prompt_tokens_total": 0.0,
                "cached_total": 0.0, "gen_throughput": 0.0}
        cur = {"generation_tokens_total": 1160.0, "prompt_tokens_total": 0.0,
               "cached_total": 0.0, "gen_throughput": 0.0}
        rates = live_metrics._rates(prev, cur, 100.0, 102.0)
        self.assertIsNotNone(rates)
        self.assertAlmostEqual(rates["decode_tok_s"], 80.0)

    def test_no_baseline_yields_none_which_is_the_empty_tile(self):
        from sglang.srt.planner import live_metrics

        self.assertIsNone(live_metrics._rates(None, {}, None, 1.0))


if __name__ == "__main__":
    unittest.main()


class TestServerInfoIsCachedNotRefetchedPerPoll(unittest.TestCase):
    """#533 part 2: the landing page aborts its own poll at LAND_POLL_MS-200
    (1800 ms). ``/get_server_info`` is served by the same app as ``/v1`` and
    awaits the scheduler, so it inherits inference back-pressure: measured
    2.97 s idle-ish and 10-35 s under a 46k-token prefill, while ``/metrics``
    on the SAME process stayed at 14 ms. Fetched per poll it blew the deadline
    every tick -- every /api/live_snapshot came back ERR_ABORTED and the page
    never left its placeholder. The data is boot-constant, so it is cached.
    """

    def setUp(self):
        from sglang.srt.planner import live_metrics

        self.lm = live_metrics
        live_metrics._SERVER_INFO_CACHE.clear()
        self.addCleanup(live_metrics._SERVER_INFO_CACHE.clear)

    def test_second_call_does_not_hit_the_network(self):
        calls = []

        def fake_get(url, timeout):
            calls.append(url)
            return {"ok": True}

        orig = self.lm._http_get_json
        self.lm._http_get_json = fake_get
        self.addCleanup(lambda: setattr(self.lm, "_http_get_json", orig))

        first = self.lm._fetch_server_info("http://x:1", 10.0)
        n_after_first = len(calls)
        second = self.lm._fetch_server_info("http://x:1", 10.0)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), n_after_first,
                         "second call re-fetched: the cache is not binding")

    def test_fetch_budget_is_capped_below_the_page_deadline(self):
        """The caller's timeout must not be able to exceed the cap -- that is
        the parameter that actually binds (a cap that never binds is reach 0).
        """
        seen = []

        def fake_get(url, timeout):
            seen.append(timeout)
            return {"ok": True}

        orig = self.lm._http_get_json
        self.lm._http_get_json = fake_get
        self.addCleanup(lambda: setattr(self.lm, "_http_get_json", orig))

        self.lm._fetch_server_info("http://y:1", 30.0)
        self.assertTrue(seen, "no fetch happened")
        self.assertLessEqual(max(seen), self.lm._SERVER_INFO_TIMEOUT_S)
        self.assertLess(max(seen), 1.8, "budget must stay under the 1800 ms "
                                        "client abort deadline")

    def test_a_failing_refresh_keeps_the_last_known_config(self):
        orig = self.lm._http_get_json
        self.lm._http_get_json = lambda url, timeout: {"ok": True}
        self.addCleanup(lambda: setattr(self.lm, "_http_get_json", orig))
        good = self.lm._fetch_server_info("http://z:1", 10.0)
        self.assertIsNotNone(good)
        # Expire the entry, then make the server unreachable.
        self.lm._SERVER_INFO_CACHE["http://z:1"] = (0.0, good)
        self.lm._http_get_json = lambda url, timeout: None
        self.assertEqual(self.lm._fetch_server_info("http://z:1", 10.0), good)
