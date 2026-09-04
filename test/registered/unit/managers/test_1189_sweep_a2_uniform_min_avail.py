"""SWEEP A2 -- ``Scheduler._uniform_min_avail`` is written by three sites,
cleared by none, and its group STOP can therefore fire at most once per
process.

RIDES WITH #1189 BECAUSE IT IS THE SAME CARDINALITY PASS. The defect-class
sweep of 2026-09-04 (`/spinning/gpu-arb/DEFECT_CLASS_SWEEP_0904.md`, charge
A2) named it beside the #1189 point fix; the ledger carries it as row
`#SWEEP-A2`. Both are round-scoped state that outlives its round.

THE CENSUS, RE-VERIFIED AT THIS TREE (`/spinning/wt-weg1`, HEAD
`6ef2c7f313`) rather than taken from the sweep -- the briefing asked for
exactly that:

    grep -n '_uniform_min_avail' python/sglang/srt/managers/scheduler.py
      6840   self._uniform_min_avail = int(kvso.dcp_min_avail())      WRITE
      6843   self._publish_uniform_evict_floor(self._uniform_min_avail)  read
      6926   self._uniform_min_avail = local_avail                    WRITE
      7181   self._uniform_min_avail = int(t[0].item())               WRITE
      8118   v = getattr(self, "_uniform_min_avail", None)            read

    three writers, two reads, ZERO clears. Confirmed: 3 writers / 0 clears,
    as the sweep measured. Nothing outside scheduler.py writes it (the nine
    other tree-wide hits are test files).

WHY THAT IS A DEFECT AND NOT A STYLE NOTE. ``uniform_min_avail()``
(`scheduler.py:8109-8141`) exists to refuse a rank-local answer to a group
question. Its refusal reads:

    "a getattr default is precisely the shape that hides such a path (#606):
     the guard reads as present in the source while being absent at runtime"

The refusal is keyed on the attribute being ABSENT. After the first
successful round the attribute is present forever, so:

  * the STOP can never fire again, however many later rounds skip the
    reduction; and
  * every such round silently returns the PREVIOUS round's group value --
    the rank-local-premise class this very guard was built to stop, arriving
    by a different door (a stale group value instead of a local one).

This is the #955 shape the sweep names in its own falsification of rule E2b:
"the only operation able to clear the flag was unreachable from the path the
flag creates". There the same shape cost 87 `#946 PREMISE RECOMPUTE` events
and 696,320 re-prefilled tokens in 14 s.

AND THE STOP IS SWALLOWED THREE TIMES OVER. All three verified verbatim at
this tree:

    scheduler.py:4624-4626   except Exception as e:  # noqa: BLE001
                             logger.warning("#888b carrier yield failed: %s", e)
                             return 0
    scheduler.py:12235-12242 except Exception as e:  # noqa: BLE001 - relief
                             must never fail a boot ... return 0
    scheduler.py:12344-12345 except Exception as e:  # noqa: BLE001
                             logger.warning("%s rung 3 (retract) failed: %s", ...)

Each wraps a call to ``self.uniform_min_avail()``. A blanket handler over a
correctness STOP is the #924 shape: a stop becomes a log line, and the run
continues on the premise the stop existed to reject.

WHAT IS *NOT* THE DEFECT, and must not be "fixed": the single-rank fallback
at `scheduler.py:8138-8141` (`if self.ps.tp_size > 1: raise` ... else return
the local `available_size()`). On one rank there is nothing to diverge from
and that is the correct answer. It has its own green test below.

CPU only, no CUDA, no distributed. Plain ``unittest.TestCase`` so a red
names its reason in the summary line.
"""

import ast
import pathlib
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

SCHEDULER = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "scheduler.py"
)

ATTR = "_uniform_min_avail"
SWALLOWERS = (4625, 12235, 12344)


def _tree():
    return ast.parse(SCHEDULER.read_text())


def _writes(tree):
    """Every ``self._uniform_min_avail = <expr>``, split into set and clear."""
    sets, clears = [], []
    for n in ast.walk(tree):
        if not isinstance(n, (ast.Assign, ast.AnnAssign)):
            continue
        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
        for t in targets:
            if (
                isinstance(t, ast.Attribute)
                and t.attr == ATTR
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                val = n.value
                is_clear = isinstance(val, ast.Constant) and val.value is None
                (clears if is_clear else sets).append(n.lineno)
    return sorted(sets), sorted(clears)


class UniformMinAvailIsRoundScoped(unittest.TestCase):
    def setUp(self):
        self.tree = _tree()

    # ---- the census, re-derived here rather than quoted -------------------

    def test_the_writer_census_is_what_the_sweep_measured(self):
        """Green today. It exists so a later change cannot move the ground.

        If this goes red the writer set changed and every claim below is
        about a different program.
        """
        sets, clears = _writes(self.tree)
        self.assertEqual(
            sets,
            [6840, 6926, 7181],
            f"the three writers of self.{ATTR} moved: {sets}",
        )

    def test_the_round_scoped_value_has_a_clear(self):
        sets, clears = _writes(self.tree)
        self.assertNotEqual(
            clears,
            [],
            f"SWEEP A2: self.{ATTR} has {len(sets)} writer(s) at {sets} and "
            f"ZERO clears in {SCHEDULER.name}. The STOP at "
            f"{SCHEDULER.name}:8131 is keyed on the attribute being ABSENT "
            f'(`getattr(self, "{ATTR}", None)` at :8118), so after the '
            f"first successful round it can never fire again, and any later "
            f"round that skipped the reduction returns the PREVIOUS round's "
            f"group value without a word. Verified independently at this "
            f"tree: `grep -n '{ATTR}' {SCHEDULER.name}` returns exactly five "
            f"lines -- :6840 :6843 :6926 :7181 :8118. The fix the sweep "
            f"names is `self.{ATTR} = None` at the head of the round that "
            f"owns it, beside the reduction in get_next_batch_to_run.",
        )

    def test_the_clear_runs_at_a_round_head(self):
        """A clear that exists but is unreachable from the round is not a fix.

        This is the #955 lesson stated as a test: there
        `pp_admission_congruence.py:789-800` had exactly one clear site, in
        the `elif entry.admitted:` arm, "unreachable from the path the flag
        creates". A textual clear check would have scored that green.
        """
        sets, clears = _writes(self.tree)
        if not clears:
            self.fail(
                f"no clear of self.{ATTR} exists at all; see "
                f"test_the_round_scoped_value_has_a_clear for the census"
            )
        fns = {}
        for n in ast.walk(self.tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for ln in clears:
                    if n.lineno <= ln <= n.end_lineno:
                        fns.setdefault(ln, []).append((n.end_lineno - n.lineno, n.name))
        owners = {ln: sorted(v)[0][1] for ln, v in fns.items()}
        self.assertTrue(
            any(
                name in ("get_next_batch_to_run", "_update_uniform_pool_budget")
                for name in owners.values()
            ),
            f"the clear(s) of self.{ATTR} sit in {owners}; none of them is "
            f"the round head that owns the reduction "
            f"(`get_next_batch_to_run` / `_update_uniform_pool_budget`). A "
            f"clear the round cannot reach leaves the STOP one-shot.",
        )

    # ---- the behaviour the census predicts -------------------------------

    def _fake(self, tp_size=3, available=1234):
        return types.SimpleNamespace(
            ps=types.SimpleNamespace(tp_size=tp_size),
            token_to_kv_pool_allocator=types.SimpleNamespace(
                available_size=lambda: available
            ),
        )

    def test_the_group_stop_can_fire_after_a_successful_round(self):
        """RED today, and this is the consequence half of the census.

        Round 1 reduces and publishes. Round 2 does not reach the reduction.
        A round-scoped value would make round 2 raise exactly as round 0 did;
        today round 2 silently answers with round 1's number.
        """
        from sglang.srt.managers.scheduler import Scheduler

        sched = self._fake()

        # Round 0: the reduce has not run. The STOP is correct here.
        with self.assertRaises(RuntimeError) as first:
            Scheduler.uniform_min_avail(sched)
        self.assertIn("_update_uniform_pool_budget", str(first.exception))

        # Round 1: the reduce ran and published the group minimum.
        sched._uniform_min_avail = 4096
        self.assertEqual(Scheduler.uniform_min_avail(sched), 4096)

        # Round 2: a new round begins and does NOT reach the reduction.
        sets, clears = _writes(self.tree)
        if not clears:
            self.fail(
                f"SWEEP A2 consequence: round 2 cannot even be expressed, "
                f"because no clear of self.{ATTR} exists ({len(sets)} "
                f"writers at {sets}, 0 clears). The STOP at "
                f"{SCHEDULER.name}:8131 therefore fires at most ONCE per "
                f"process: after round 1 the attribute is present forever, "
                f"and every later round that skips the reduction returns "
                f"round 1's group value -- 4096 here -- as though it were "
                f"this round's. That is the rank-local-premise class the "
                f"guard's own comment cites #606 against, arriving through a "
                f"stale group value instead of a local one."
            )
        # A clear exists. Drive it if it is exposed as a callable; otherwise
        # it is an inline statement in the round head -- whose PLACEMENT is
        # asserted by test_the_clear_runs_at_a_round_head -- and the round
        # boundary is modelled here. Deliberately NOT a demand for a
        # particular method name: an inline `self._uniform_min_avail = None`
        # at the head of get_next_batch_to_run is a correct fix and must not
        # be scored red by this suite.
        clear_fn = getattr(Scheduler, "clear_round_scoped_uniform_budget", None)
        if clear_fn is None:
            del sched._uniform_min_avail
        else:
            clear_fn(sched)
        with self.assertRaises(RuntimeError):
            Scheduler.uniform_min_avail(sched)

    # ---- the three swallowers --------------------------------------------

    def test_the_three_swallowers_reraise_the_group_stop(self):
        """A blanket ``except Exception`` over a correctness STOP.

        All three wrap a call to ``self.uniform_min_avail()`` and turn its
        RuntimeError into a warning plus ``return 0`` / a continued ladder.
        The sweep's prescription is narrow and is what this test encodes:
        keep swallowing everything else, re-raise THIS one.
        """
        offenders = []
        for n in ast.walk(self.tree):
            if not isinstance(n, ast.Try):
                continue
            calls_uniform = any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "uniform_min_avail"
                for c in ast.walk(ast.Module(body=n.body, type_ignores=[]))
            )
            if not calls_uniform:
                continue
            for h in n.handlers:
                blanket = h.type is None or (
                    isinstance(h.type, ast.Name) and h.type.id == "Exception"
                )
                if not blanket:
                    continue
                reraises = any(
                    isinstance(x, ast.Raise)
                    for x in ast.walk(ast.Module(body=h.body, type_ignores=[]))
                )
                if not reraises:
                    offenders.append(h.lineno)
        self.assertEqual(
            sorted(offenders),
            [],
            f"SWEEP A2: blanket handlers at {SCHEDULER.name}:"
            f"{sorted(offenders)} wrap a call to uniform_min_avail() and "
            f"contain no `raise`, so the group STOP at :8131 becomes a "
            f"logger.warning and the round proceeds on the premise the STOP "
            f"exists to reject. Expected offenders at this tree: "
            f"{list(SWALLOWERS)} (:4625 #888b carrier yield, :12235 the "
            f"ladder trigger, :12344 rung 3 retract). Narrow fix: re-raise "
            f"this RuntimeError, keep swallowing the rest.",
        )

    # ---- controls that are GREEN today and must stay green ---------------

    def test_single_rank_fallback_is_preserved(self):
        """`scheduler.py:8138-8141`. NOT the defect -- do not remove it.

        On one rank there is nothing to diverge from, so the live local
        `available_size()` is the correct answer and the STOP must not fire.
        """
        from sglang.srt.managers.scheduler import Scheduler

        sched = self._fake(tp_size=1, available=777)
        self.assertEqual(Scheduler.uniform_min_avail(sched), 777)

    def test_a_published_group_value_is_returned_not_the_local_one(self):
        """The reduced value outranks the local pool. Green today and after."""
        from sglang.srt.managers.scheduler import Scheduler

        sched = self._fake(tp_size=3, available=999999)
        sched._uniform_min_avail = 42
        self.assertEqual(Scheduler.uniform_min_avail(sched), 42)


if __name__ == "__main__":
    unittest.main()
