"""#867 guard: a broad handler around a device op must not swallow a POISONED context.

THE CLASS, found four times in one night by MIS-ATTRIBUTION and never once by
inspection: a handler that catches BREADTH instead of KIND, making two states
indistinguishable -- "the poll hiccuped" and "the CUDA context is dead".

A CUDA illegal memory access poisons the context. Every later CUDA call in the
process then raises the SAME error at whatever site runs next. So

    except Exception:                      # "a watchdog must not die"
        logger.exception("... failed")

does not keep the process alive. It only decides that the crash will be
reported somewhere innocent. W40 produced THREE different innocent crash sites
for one fault across three boots (seam restore, seam capture, prefill batch
build) before ``barlink_abort_gate`` was taught the difference.

WHY A GUARD AND NOT ANOTHER FIX. The sweep that found this class turned up ~20
qualifying handlers and could not honestly claim to be exhaustive -- roughly 117
CUDA-adjacent broad handlers exist and nobody has read them all. A guard names
them mechanically, so the gap closes without anyone reading 117 handlers, and it
keeps naming them as new ones are written. That is the difference between fixing
an instance and closing a class.

SCOPE, DELIBERATELY NARROW, because a guard that fires on every harmless
``except Exception`` in the tree is switched off within a week. A handler is
reported only when ALL THREE hold:

  1. the handler is BROAD -- bare ``except:``, ``except Exception``, or
     ``except BaseException``;
  2. the guarded body reaches a DEVICE operation (see ``_DEVICE_TOKENS``);
  3. it sits in a REPEATING context -- inside a loop, or in a function whose
     name marks it as a poller/worker/tick -- because those swallow the same
     poison again and again, which is what makes the class expensive.

A handler is COVERED when it consults ``is_poison_error`` or re-raises.

KNOWN LIMIT, STATED RATHER THAN HIDDEN: condition 2 is a ONE-HOP name test. A
handler whose try body calls a helper that only deeper down touches the device
is invisible to it unless that helper's name is in ``_DEVICE_TOKENS``. This
guard therefore proves "these sites are uncovered", never "no others exist".
The unread files named in the #867 sweep -- barlink_bar1.py:4192/4854/4903/5898,
barlink_matrix.py, barlink_ucx.py, mem_cache/kv_vmm_backing.py -- are covered by
it only to the extent their handlers meet condition 2 directly.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import ast
import pathlib
import tempfile
import unittest

from sglang.test.test_utils import CustomTestCase

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PKG = _REPO_ROOT / "python" / "sglang" / "srt"

#: Tokens whose presence in a guarded body means "this can touch the device".
#: Attribute names and bare names both count. Kept tight on purpose: every entry
#: here is a real device op or a known device poller in this tree.
_DEVICE_TOKENS = frozenset(
    {
        "synchronize",
        "record_stream",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "empty_cache",
        "mem_get_info",
        "nccl",
        "cuda",
        # Known device pollers whose own bodies reach the device one or more
        # hops down. Extend this list when a new one is written -- that is the
        # maintenance cost of the one-hop limit named in the module docstring.
        "poll_status_word",
        "poll_status_words",
        "poll_abort_words",
        "probe_once",
    }
)

#: Function names that mark a repeating context even without a literal loop.
_REPEATING_NAMES = ("loop", "tick", "poll", "watchdog", "monitor", "worker",
                    "heartbeat", "probe", "_run")

_COVERED_TOKENS = frozenset({"is_poison_error", "record_poison"})

#: Sites known to be uncovered, each a deliberate decision to defer -- NOT a
#: verdict that they are safe. THIS LIST MUST ONLY EVER SHRINK. Adding to it is
#: how a class quietly reopens, so a new entry needs the same argument the two
#: removals below carried.
#:
#: Already removed by consulting the classifier:
#:   barlink_liveness.py  the outer swallow around the very function #867 fixed
#:   dual_group_lane.py   a daemon thread that relaunched kernels after a fault
#:
#: Three of these were in files the #867 sweep listed as UNREAD
#: (barlink_bar1.py:4903, barlink_ucx.py:1964/1972). The guard found them
#: without anyone reading 117 handlers, which is the point of building it.
_ACCEPTED: frozenset = frozenset(
    {
        "python/sglang/srt/debug_utils/spec_state_hash.py:356",
        "python/sglang/srt/disaggregation/encode_server.py:2112",
        "python/sglang/srt/distributed/device_communicators/barlink.py:598",
        "python/sglang/srt/distributed/device_communicators/barlink_bar1.py:4903",
        "python/sglang/srt/distributed/device_communicators/barlink_device.py:1469",
        "python/sglang/srt/distributed/device_communicators/barlink_ucx.py:1964",
        "python/sglang/srt/distributed/device_communicators/barlink_ucx.py:1972",
        "python/sglang/srt/managers/cache_controller.py:1340",
        "python/sglang/srt/managers/kv_session_spill_destination.py:914",
        "python/sglang/srt/managers/phase_flip_seam_census.py:174",
        "python/sglang/srt/managers/scheduler.py:1619",
        "python/sglang/srt/utils/common.py:724",
    }
)


def _tokens(node) -> set:
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
    return out


def _is_broad(handler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in ("Exception", "BaseException")
    return False


def _covered(handler, poison_aware: frozenset = frozenset()) -> bool:
    """Does this handler ask the KIND question, or refuse to continue?

    ``poison_aware`` carries the names of helpers that consult the classifier
    themselves, resolved one hop -- symmetric with the one-hop device test
    above. Without it, extracting the check into a well-named helper (which is
    what a second handler in the same class should do) would read as UNCOVERED,
    and the guard would push people toward copy-paste. That defect was live in
    this guard's own first run: it reported `barlink_liveness`'s two FIXED
    handlers because they delegate to `_stop_on_poison`.
    """
    if any(isinstance(s, ast.Raise) for s in ast.walk(handler)):
        return True
    tokens = _tokens(handler)
    return bool(tokens & _COVERED_TOKENS) or bool(tokens & poison_aware)


def _poison_aware_names(root: pathlib.Path) -> frozenset:
    """Functions whose own body consults the classifier."""
    names = set()
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _tokens(node) & _COVERED_TOKENS:
                    names.add(node.name)
    return frozenset(names)


def _repeating_ranges(tree) -> list:
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.While, ast.For, ast.AsyncFor)):
            spans.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            m in node.name.lower() for m in _REPEATING_NAMES
        ):
            spans.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
    return spans


def _is_test_path(path: pathlib.Path) -> bool:
    parts = path.parts
    return "test" in parts or path.name.startswith("test_")


def sweep_swallowed_poison(root: pathlib.Path, repo_root: pathlib.Path = None):
    """``(uncovered, candidates)``.

    ``uncovered`` maps ``"relpath:lineno"`` to the guarded body's device token,
    so a failure names the site AND why the guard thinks it touches the device.
    """
    repo_root = repo_root or _REPO_ROOT
    poison_aware = _poison_aware_names(root)
    uncovered = {}
    candidates = 0
    for path in sorted(root.rglob("*.py")):
        if repo_root is _REPO_ROOT and _is_test_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
        except SyntaxError:
            continue
        rel = path.relative_to(repo_root).as_posix()
        spans = _repeating_ranges(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            body_tokens = set()
            for stmt in node.body:
                body_tokens |= _tokens(stmt)
            hit = body_tokens & _DEVICE_TOKENS
            if not hit:
                continue
            for handler in node.handlers:
                if not _is_broad(handler):
                    continue
                if not any(lo <= handler.lineno <= hi for lo, hi in spans):
                    continue
                candidates += 1
                if _covered(handler, poison_aware):
                    continue
                uncovered[f"{rel}:{handler.lineno}"] = sorted(hit)[0]
    return uncovered, candidates


class TestTheTreeHasNoSwallowedPoison(CustomTestCase):
    def test_the_guard_actually_swept_something(self):
        """A guard that matches nothing passes for the wrong reason."""
        _, candidates = sweep_swallowed_poison(_PKG)
        self.assertGreater(
            candidates,
            5,
            "the guard found almost no broad handlers around device ops, which "
            "means its own matching broke, not that the tree is clean",
        )

    def test_no_broad_handler_swallows_a_poisoned_context(self):
        uncovered, _ = sweep_swallowed_poison(_PKG)
        offenders = {k: v for k, v in uncovered.items() if k not in _ACCEPTED}
        self.assertEqual(
            offenders,
            {},
            "these broad handlers sit in a repeating context around a device "
            "op and never ask whether the error poisoned the CUDA context. "
            "Consult barlink_abort_gate.is_poison_error and stop, or re-raise; "
            "swallowing decides that the crash is reported somewhere innocent",
        )


class TestTheGuardCanFailInBothDirections(CustomTestCase):
    """The half that keeps the guard honest: it must go RED on a planted
    offender and must stay SILENT on the harmless shapes around it."""

    def _tree(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        for name, body in files.items():
            (root / name).write_text(body)
        return root

    _OFFENDER = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def poll_loop(dev):\n"
        "    while True:\n"
        "        try:\n"
        "            dev.synchronize()\n"
        "        except Exception:\n"
        "            logger.exception('failed')\n"
    )

    def test_it_reports_a_planted_swallowed_handler(self):
        root = self._tree({"bad.py": self._OFFENDER})
        uncovered, candidates = sweep_swallowed_poison(root, repo_root=root)
        self.assertEqual(candidates, 1)
        self.assertEqual(sorted(uncovered), ["bad.py:7"])
        self.assertEqual(uncovered["bad.py:7"], "synchronize")

    def test_it_goes_silent_once_the_handler_consults_the_classifier(self):
        fixed = self._OFFENDER.replace(
            "        except Exception:\n            logger.exception('failed')\n",
            "        except Exception as exc:\n"
            "            if is_poison_error(exc):\n"
            "                break\n"
            "            logger.exception('failed')\n",
        )
        root = self._tree({"bad.py": fixed})
        uncovered, candidates = sweep_swallowed_poison(root, repo_root=root)
        self.assertEqual(candidates, 1, "still a candidate, now covered")
        self.assertEqual(uncovered, {})

    def test_a_reraise_also_counts_as_covered(self):
        fixed = self._OFFENDER.replace(
            "            logger.exception('failed')\n",
            "            logger.exception('failed')\n            raise\n",
        )
        root = self._tree({"bad.py": fixed})
        uncovered, _ = sweep_swallowed_poison(root, repo_root=root)
        self.assertEqual(uncovered, {})

    def test_a_narrow_handler_is_not_reported(self):
        """CONTROL: catching a NAMED error is the opposite of this defect."""
        narrow = self._OFFENDER.replace(
            "        except Exception:\n", "        except TimeoutError:\n"
        )
        root = self._tree({"bad.py": narrow})
        uncovered, candidates = sweep_swallowed_poison(root, repo_root=root)
        self.assertEqual((uncovered, candidates), ({}, 0))

    def test_a_broad_handler_with_no_device_op_is_not_reported(self):
        """CONTROL, and the one that keeps the guard from being switched off:
        the tree is full of harmless broad handlers and none may fire."""
        harmless = (
            "import logging\n"
            "logger = logging.getLogger(__name__)\n"
            "def poll_loop(cfg):\n"
            "    while True:\n"
            "        try:\n"
            "            cfg.reload_from_disk()\n"
            "        except Exception:\n"
            "            logger.exception('failed')\n"
        )
        root = self._tree({"ok.py": harmless})
        uncovered, candidates = sweep_swallowed_poison(root, repo_root=root)
        self.assertEqual((uncovered, candidates), ({}, 0))

    def test_a_one_shot_handler_outside_a_loop_is_not_reported(self):
        """CONTROL: the class is about REPEATED swallowing. A single
        best-effort call at startup is a different judgement and not this one."""
        one_shot = (
            "import logging\n"
            "logger = logging.getLogger(__name__)\n"
            "def setup(dev):\n"
            "    try:\n"
            "        dev.synchronize()\n"
            "    except Exception:\n"
            "        logger.exception('failed')\n"
        )
        root = self._tree({"once.py": one_shot})
        uncovered, candidates = sweep_swallowed_poison(root, repo_root=root)
        self.assertEqual((uncovered, candidates), ({}, 0))


if __name__ == "__main__":
    unittest.main()
