"""#274 round 8: the lane-spec window driver, without a card.

A measurement driver is an instrument, and an instrument that is only ever
exercised on the rig is calibrated by the thing it measures. Every rule this
driver applies -- the ring-safe poll, the void-vs-divergent verdict, the
ms/token normalisation, the card-equivalent arithmetic, the queue depth --
is decidable against a fake server, and each of them has cost real card time
at least once when it was not.

The fake server is deliberately faithful in the one property that bit before:
``results`` is a RING of the last eight rows while ``results_total`` is
monotone.
"""

import importlib.util
import pathlib
import sys
import threading
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DUAL_GROUP = _REPO_ROOT / "scripts" / "dual_group"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if str(_DUAL_GROUP) not in sys.path:
    sys.path.insert(0, str(_DUAL_GROUP))

probe = _load("lane_accept_probe", _DUAL_GROUP / "lane_accept_probe.py")
window = _load("lane_spec_window", _DUAL_GROUP / "r8" / "lane_spec_window.py")


class FakeServer:
    """Enough of ``/get_server_info`` and ``/set_internal_state`` to drive the
    poll loop, with the result ring the real lane has."""

    RING = 8

    def __init__(self, out_ids=None):
        self.results = []
        self.results_total = 0
        self.queued = 0
        self.active = False
        self.decode_tokens = 0
        self.posted = []
        self.out_ids = out_ids or (lambda job: [1, 2, 3])
        self._lock = threading.Lock()

    # -- transport ------------------------------------------------------
    def get(self, base, path, timeout=None):
        with self._lock:
            return {
                "internal_states": [
                    {
                        "dual_group_lanes": [
                            {
                                "lane_id": 0,
                                "queued": self.queued,
                                "active": self.active,
                                "results_total": self.results_total,
                                "results": list(self.results[-self.RING :]),
                                "work_total": {"decode_tokens": self.decode_tokens},
                            }
                        ]
                    }
                ]
            }

    def post(self, base, path, payload, timeout=None):
        if path == "/set_internal_state":
            job = payload["server_args"]["dual_group_lane_prefill"]
            with self._lock:
                self.posted.append(job)
                self.complete(job)
            return {}
        if path == "/generate":
            return {"meta_info": {"completion_tokens": 7}}
        raise AssertionError(path)

    # -- lane behaviour --------------------------------------------------
    def complete(self, job):
        ids = self.out_ids(job)
        self.results.append(
            {
                "spec_mode": bool(job.get("spec")),
                "decode_steps": len(ids),
                "spec_rounds": len(ids) if job.get("spec") else None,
                "accept_len_mean": 1.0 if job.get("spec") else None,
                "decode_ms_mean": 24.0 if job.get("spec") else 16.0,
                "verify_graph_rounds": len(ids) if job.get("spec") else None,
                "output_ids": ids,
            }
        )
        self.results_total += 1
        self.decode_tokens += len(ids)


#: A scorable id -> letter map for the stubbed detokenizer.
#:
#: The gate grades the four arms since the r12 control (#284), so a test that
#: drives it with opaque ids has to say what those ids MEAN as text -- an
#: identity check needed no tokenizer, a graded one does. The map is chosen so
#: the `alphabet` scorer can see a difference: 7/8/9/10 are the determined
#: tail w/x/y/z and 99 is a wrong letter.
_ID_TEXT = {7: "w", 8: "x", 9: "y", 10: "z", 99: "q"}


def _fake_detokenize(ids, tokenizer_path):
    return "\n".join(_ID_TEXT.get(i, "?") for i in ids)


class _Patched:
    def __init__(self, server):
        self.server = server
        self._saved = {}

    def __enter__(self):
        for mod in (probe, window):
            self._saved[(mod, "_get")] = getattr(mod, "_get", None)
            self._saved[(mod, "_post")] = getattr(mod, "_post", None)
            mod._get = self.server.get
            mod._post = self.server.post
        self._saved[(window, "detokenize")] = getattr(window, "detokenize", None)
        window.detokenize = _fake_detokenize
        return self.server

    def __exit__(self, *exc):
        for (mod, attr), value in self._saved.items():
            if value is not None:
                setattr(mod, attr, value)
        return False


class TestRingSafePoll(CustomTestCase):
    """The trap that cost 15 minutes of a card window, pinned.

    ``results`` holds the last eight rows. A poll on its LENGTH stops
    growing at the ninth job of a boot and every later job then waits out
    its whole budget on work that already finished.
    """

    def test_a_boot_can_run_more_jobs_than_the_ring_holds(self):
        server = FakeServer()
        with _Patched(server):
            for i in range(12):
                rows = probe.lane_run(
                    "http://x",
                    {"lane_id": 0, "input_ids": [1], "max_new_tokens": 3},
                    poll_s=0.0,
                    budget_s=2.0,
                )
                self.assertTrue(rows, f"job {i} returned nothing")
        self.assertEqual(server.results_total, 12)
        self.assertEqual(len(server.results), 12)  # the fake keeps all of them
        # ... but the endpoint only ever exposes the last eight.
        exposed = server.get("", "")["internal_states"][0]["dual_group_lanes"][0]
        self.assertEqual(len(exposed["results"]), FakeServer.RING)

    def test_the_poll_reads_the_monotone_counter_not_the_ring(self):
        server = FakeServer()
        with _Patched(server):
            total, rows = probe.lane_snapshot("http://x")
            self.assertEqual((total, rows), (0, []))
            server.complete({"input_ids": [1]})
            total, rows = probe.lane_snapshot("http://x")
        self.assertEqual(total, 1)
        self.assertEqual(len(rows), 1)


class TestGateVerdict(CustomTestCase):
    """Coherent, divergent and VOID are three outcomes, not two.

    A prompt whose own A-vs-A floor is not byte-identical carries no verdict
    in either direction: the instrument did not hold still, so a difference
    between the arms cannot be attributed to the arms. Round 4 measured one
    of four candidate prompts failing exactly that check.
    """

    def _run(self, out_ids):
        server = FakeServer(out_ids=out_ids)
        with _Patched(server):
            window.tokenize = lambda base, text, tok: [1, 2, 3]
            return window.run_gate(
                "http://x", "tok", ["alphabet"], 4, 1, "target_verify", 1e18
            )

    def test_identical_arms_are_coherent(self):
        got = self._run(lambda job: [7, 8, 9])
        self.assertEqual(got["verdict"], "coherent")
        self.assertTrue(got["prompts"]["alphabet"]["spec_matches_no_spec"])
        self.assertIsNone(got["prompts"]["alphabet"]["first_divergent_index"])

    def test_a_speculative_arm_that_scores_worse_is_divergent_and_says_where(self):
        # w x q instead of w x y: the divergence costs a point, so it fails.
        def out_ids(job):
            return [7, 8, 99] if job.get("spec") else [7, 8, 9]

        got = self._run(out_ids)
        self.assertEqual(got["verdict"], "divergent")
        self.assertEqual(got["prompts"]["alphabet"]["first_divergent_index"], 2)
        self.assertEqual(got["prompts"]["alphabet"]["graded_delta"], -1)

    def test_a_divergence_that_costs_nothing_is_coherent(self):
        """The r12 control, as a rule (#284 -> docs/dev/ANALYSE_284...).

        Stock NEXTN leaves the stock greedy trajectory on this vehicle with no
        lane in the process, at the smallest top-2 margin of the run and at an
        identical graded score. A gate that failed on that was measuring the
        margin at one position. Different tokens, same score -> coherent, and
        the token-identity rate says the tokens differed.
        """
        # w x y z then a flip: the determined tail is complete on both arms,
        # so the flip lands where the task no longer determines an answer --
        # which is exactly where the rig measured it (index 63 of 64, after
        # the alphabet had already ended).
        def out_ids(job):
            return [7, 8, 9, 10, 8] if job.get("spec") else [7, 8, 9, 10, 99]

        got = self._run(out_ids)
        prompt = got["prompts"]["alphabet"]
        self.assertEqual(prompt["classification"], "content_divergence")
        self.assertEqual(prompt["first_divergent_index"], 4)
        self.assertEqual(prompt["graded_delta"], 0)
        self.assertTrue(prompt["graded_within_band"])
        self.assertEqual(got["verdict"], "coherent")
        self.assertEqual(got["token_identity_rate"], 0.0)

    def test_one_extra_token_at_the_end_is_not_a_divergence(self):
        # A speculative round emits accept+1 tokens as a BLOCK, so the last
        # block can carry the job past max_new_tokens. Measured on the rig
        # (round 8, `repeat`): 64 identical tokens, one extra. Calling that a
        # divergence turns a clean gate red and hides the one prompt that
        # really did diverge in the same run.
        def out_ids(job):
            return [7, 8, 9, 10] if job.get("spec") else [7, 8, 9]

        got = self._run(out_ids)
        prompt = got["prompts"]["alphabet"]
        self.assertEqual(prompt["classification"], "length_end_only")
        self.assertTrue(prompt["shared_prefix_identical"])
        self.assertIsNone(prompt["first_divergent_index"])
        self.assertEqual(got["verdict"], "coherent")

    def test_an_unstable_floor_voids_the_prompt_rather_than_failing_it(self):
        seen = {"n": 0}

        def out_ids(job):
            seen["n"] += 1
            return [7, 8, seen["n"]]

        got = self._run(out_ids)
        self.assertFalse(got["prompts"]["alphabet"]["floor_byte_identical"])
        self.assertIn("void", got["prompts"]["alphabet"])
        self.assertEqual(got["verdict"], "void")
        self.assertEqual(got["judged_prompts"], 0)

    def test_a_stable_no_spec_floor_does_not_certify_the_speculative_side(self):
        # The round-8 finding, pinned. On `alphabet` both no-spec runs agreed
        # (they take the identical kernel path) while two runs of the IDENTICAL
        # speculative arm disagreed with each other in one boot. Read against
        # the no-spec floor alone that looks like a chain defect; it is a
        # position whose logit margin is under the numeric difference between
        # a 1-row decode and a 2-row verify.
        seen = {"n": 0}

        def out_ids(job):
            if not job.get("spec"):
                return [7, 8, 9]
            seen["n"] += 1
            return [7, 8, 9] if seen["n"] == 1 else [7, 99, 9]

        got = self._run(out_ids)
        prompt = got["prompts"]["alphabet"]
        self.assertTrue(prompt["floor_byte_identical"])
        self.assertFalse(prompt["spec_floor_byte_identical"])
        self.assertIn("speculative a-vs-a floor", prompt["void"])
        self.assertEqual(got["verdict"], "void")
        self.assertEqual(got["judged_prompts"], 0)

    def test_a_reproducible_divergence_that_costs_score_fails_the_gate(self):
        # The counterpart: BOTH floors hold, the arms disagree AND the answer
        # got worse. Nothing explains that away and the gate is red.
        def out_ids(job):
            return [7, 99, 9] if job.get("spec") else [7, 8, 9]

        got = self._run(out_ids)
        prompt = got["prompts"]["alphabet"]
        self.assertTrue(prompt["floor_byte_identical"])
        self.assertTrue(prompt["spec_floor_byte_identical"])
        self.assertEqual(got["verdict"], "divergent")
        self.assertEqual(prompt["first_divergent_index"], 1)

    def test_the_three_classes_are_decided_on_the_shared_prefix(self):
        cmp = window.compare_trajectories
        self.assertEqual(cmp([1, 2, 3], [1, 2, 3])["classification"], "identical")
        self.assertEqual(
            cmp([1, 2, 3], [1, 2, 3, 4])["classification"], "length_end_only"
        )
        self.assertEqual(
            cmp([1, 2, 3, 4], [1, 2, 3])["classification"], "length_end_only"
        )
        got = cmp([1, 2, 3, 4], [1, 9, 3, 4])
        self.assertEqual(got["classification"], "content_divergence")
        self.assertEqual(got["first_divergent_index"], 1)

    def test_the_gate_stops_at_its_deadline_and_says_so(self):
        server = FakeServer()
        with _Patched(server):
            window.tokenize = lambda base, text, tok: [1, 2, 3]
            got = window.run_gate(
                "http://x", "tok", ["alphabet", "squares"], 4, 1, "target_verify", 0.0
            )
        self.assertEqual(got["truncated_at"], "alphabet")
        self.assertEqual(got["prompts"], {})


class TestMsPerToken(CustomTestCase):
    """A speculative round is slower AND emits more than one token.

    Comparing round times alone always flatters the non-speculative arm, so
    the driver normalises to the tokens the rounds actually produced.
    """

    def test_a_non_speculative_arm_is_its_own_round_time(self):
        self.assertEqual(
            window._ms_per_token({"round_ms_mean": 16.1, "spec_rounds": None}), 16.1
        )

    def test_a_speculative_arm_is_divided_by_what_it_emitted(self):
        # 24 ms per round, 10 rounds, 16 tokens -> 15.0 ms/token, i.e. a win
        # against a 16.1 ms no-spec step even though the round is slower.
        got = window._ms_per_token(
            {"round_ms_mean": 24.0, "spec_rounds": 10, "output_ids": list(range(16))}
        )
        self.assertEqual(got, 15.0)

    def test_decode_steps_is_not_the_denominator(self):
        # For a speculative job the lane appends one decode_ms entry per ROUND,
        # so decode_steps == spec_rounds and dividing by it returns the round
        # time unchanged -- the normalisation would silently do nothing. This
        # row has both fields and they disagree on purpose.
        got = window._ms_per_token(
            {
                "round_ms_mean": 24.938,
                "spec_rounds": 45,
                "decode_steps": 45,
                "output_ids": list(range(64)),
            }
        )
        self.assertEqual(got, 17.535)
        self.assertNotEqual(got, 24.938)

    def test_a_round_that_emitted_nothing_reports_nothing(self):
        self.assertIsNone(
            window._ms_per_token(
                {"round_ms_mean": 24.0, "spec_rounds": 10, "output_ids": []}
            )
        )


class TestCardEquivalents(CustomTestCase):
    """E = share_serving + share_lane, on the wall clock, per §11.5."""

    def _fake_windows(self, rows):
        it = iter(rows)

        def _window(base, window_s, serving, lane):
            return next(it)

        return _window

    def test_E_is_the_sum_of_the_two_shares(self):
        saved = window._window
        window._window = self._fake_windows(
            [
                {"serving_tok_s": 40.0, "lane_tok_s": None},
                {"serving_tok_s": 0, "lane_tok_s": 10.0},
                {"serving_tok_s": 35.2, "lane_tok_s": 10.02},
            ]
        )
        window.tokenize = lambda base, text, tok: [1, 2, 3]
        try:
            got = window.run_card_equivalents(
                "http://x",
                "tok",
                "squares",
                1.0,
                8,
                8,
                2,
                1,
                "target_verify",
                True,
                1e18,
            )
        finally:
            window._window = saved
        self.assertEqual(got["share_serving"], 0.88)
        self.assertEqual(got["share_lane"], 1.002)
        self.assertEqual(got["E"], 1.882)

    def test_a_zero_solo_floor_yields_no_share_rather_than_a_division(self):
        self.assertIsNone(window._ratio(1.0, 0.0))
        self.assertIsNone(window._ratio(None, 1.0))

    def test_the_phase_is_skipped_whole_rather_than_cut_mid_window(self):
        window.tokenize = lambda base, text, tok: [1, 2, 3]
        got = window.run_card_equivalents(
            "http://x",
            "tok",
            "squares",
            1.0,
            8,
            8,
            2,
            1,
            "target_verify",
            True,
            0.0,
        )
        self.assertEqual(got["skipped"], "deadline")
        self.assertNotIn("solo_serving", got)


class TestLaneLoadDepth(CustomTestCase):
    """The feeder holds a shallow QUEUE, it does not run jobs back to back.

    A feeder that waits for each job leaves the lane idle for one poll per
    job. That idle time is a larger fraction of a fast solo window than of a
    slow shared one, so it biases exactly the ratio the phase computes.
    """

    def test_the_feeder_tops_the_queue_up_to_its_depth(self):
        server = FakeServer()
        with _Patched(server):
            load = window.LaneLoad("http://x", [1, 2], 4, True, 1, "target_verify")
            # One pass of the top-up logic, without the thread.
            snap = window._lane_snapshot("http://x")
            depth = int(snap.get("queued") or 0) + (1 if snap.get("active") else 0)
            for _ in range(max(0, load.DEPTH - depth)):
                load._post_one()
        self.assertEqual(load.posted, window.LaneLoad.DEPTH)

    def test_a_busy_lane_is_not_topped_up_past_the_depth(self):
        server = FakeServer()
        server.queued = 1
        server.active = True
        with _Patched(server):
            load = window.LaneLoad("http://x", [1, 2], 4, True, 1, "target_verify")
            snap = window._lane_snapshot("http://x")
            depth = int(snap.get("queued") or 0) + (1 if snap.get("active") else 0)
            for _ in range(max(0, load.DEPTH - depth)):
                load._post_one()
        self.assertEqual(load.posted, 0)

    def test_wait_lane_idle_returns_false_instead_of_blocking_forever(self):
        server = FakeServer()
        server.queued = 3
        with _Patched(server):
            self.assertFalse(window.wait_lane_idle("http://x", budget_s=0.2))

    def test_wait_lane_idle_returns_true_on_a_drained_lane(self):
        server = FakeServer()
        with _Patched(server):
            self.assertTrue(window.wait_lane_idle("http://x", budget_s=1.0))


if __name__ == "__main__":
    unittest.main()
