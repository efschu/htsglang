"""Guard for the behaviour-neutral split of ``scheduler.py`` (27.10.): methods moved into ``Scheduler*Mixin`` classes.

Three silent failure modes the split opens, each of which every other test would pass through:

* a method defined twice (in ``Scheduler`` and a mixin, or in two mixins): the one earlier in the MRO wins and the other is
  dead code that still reads like the live one;
* a split mixin placed AFTER one of the older mixins in the bases: a name both define would silently resolve to the older one;
* a mixin file that exists but never reaches ``Scheduler``'s bases: its methods vanish without an import error.
"""

import ast
import pathlib

from flliper.srt.managers import scheduler as sched

_MANAGERS = pathlib.Path(sched.__file__).parent
_SPLIT_FILES = sorted(p for p in _MANAGERS.glob("scheduler_*_mixin.py") if p.name != "scheduler_pp_mixin.py")


def _split_classes():
    out = []
    for path in _SPLIT_FILES:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Scheduler") and node.name.endswith("Mixin"):
                out.append((path.name, node.name, {m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}))
    return out


def test_every_split_mixin_file_reaches_the_scheduler_bases():
    names = {n for _, n, _ in _split_classes()}
    assert names, "no split mixin found: the glob or the naming drifted"
    assert names <= {c.__name__ for c in sched.Scheduler.__mro__}, names - {c.__name__ for c in sched.Scheduler.__mro__}


def test_a_method_is_defined_in_exactly_one_split_class():
    seen = {}
    own = {m.name for m in ast.parse(pathlib.Path(sched.__file__).read_text(encoding="utf-8")).body
           if isinstance(m, ast.ClassDef) and m.name == "Scheduler" for m in m.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for fname, cname, methods in _split_classes():
        for m in methods:
            assert m not in own, f"{m} is defined in Scheduler and in {cname} ({fname})"
            assert m not in seen, f"{m} is defined in {cname} and in {seen[m]}"
            seen[m] = cname


def test_the_split_mixins_come_before_every_older_mixin_in_the_mro():
    mro = [c.__name__ for c in sched.Scheduler.__mro__]
    split = {n for _, n, _ in _split_classes()}
    last_split = max(mro.index(n) for n in split)
    older = [n for n in mro[1:-1] if n not in split]
    assert older, "the older mixins vanished from the MRO"
    assert all(mro.index(n) > last_split for n in older), mro
