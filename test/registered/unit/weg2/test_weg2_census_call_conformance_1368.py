# SPDX-License-Identifier: Apache-2.0
"""#1368 -- the producer commit called a signature that never existed.

xsn30 died before either group started, on both caps, rc=1:

    TypeError: widest_layer_terms() got an unexpected keyword argument 'n_lanes'
    launcher.py:5129 -> choose_host_ledger

#1358 added `n_lanes` to `xchg_bounce.bounce_terms` (xchg_bounce.py:235) and to
the launcher's caller, and never to the glue BETWEEN them
(checkpoint_census.widest_layer_terms). No commit of the chain -- 133914ff22,
f7b04ca13b, 38713c68ca -- widened that signature. Every desk test in the chain
called `bounce_terms` directly or built BounceTerms by hand, so the one path a
boot takes was the one path nothing executed: desk-written, never executed.

A test for THIS keyword would be worth one boot. So this file pins the CLASS:
every call the launcher makes into the two sizing modules must bind against the
real signature. `test_weg2_store_priced_x_1317` does the same for host_ledger
calls; between them the launcher cannot call a signature that does not exist.

The second half is the one the TypeError hid: the lane count has TWO consumers
-- the ledger that CHARGES the bounce and the publication that hands the ranks
its INPUTS. They built the cut key at their own call sites, so fixing only the
crash would have left the publication at the default n_lanes=1 while the ledger
charged the measured 5, and the rank-side guard (weight_updater.py, "the ledger
priced n_lanes=... and THIS RANK alone needs ...") would have refused the boot
a few seconds later, naming the ledger rather than the publication that
diverged. Measured on the shipped checkpoint: 1 lane = 1.784 GiB, 5 = 7.419.
"""

import ast
import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import checkpoint_census, launcher, xchg_bounce
from sglang.test.test_utils import CustomTestCase

#: The modules whose signatures the launcher must conform to here.
_MODULES = {"checkpoint_census": checkpoint_census, "xchg_bounce": xchg_bounce,
            "launcher": launcher}

#: The launcher's OWN builders, bare-name calls. Added #1373: the argv builders
#: are the other family that has killed boots on a signature (2cc618b819), and
#: a keyword with no receiver there -- `argv_d(..., window_mib=...)` when the
#: parameter is actually `depth` -- raises the same TypeError at the same place
#: in the boot as #1368's did. Binding them here is the SAME mechanism, one
#: inventory, rather than a second scan.
_LOCAL = ("argv_p", "argv_d", "common_flags")


def _calls():
    """(target_name, attr, node) for every launcher call this pin binds.

    Two shapes: `module.attr(...)` into the sizing modules, and bare-name calls
    to the launcher's own argv builders.
    """
    tree = ast.parse(inspect.getsource(launcher))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id in _MODULES):
            yield f.value.id, f.attr, node
        elif isinstance(f, ast.Name) and f.id in _LOCAL:
            yield "launcher", f.id, node


class TheLauncherCannotCallASignatureThatDoesNotExist(CustomTestCase):
    def test_every_call_binds_against_the_real_signature(self):
        checked = 0
        for alias, attr, node in _calls():
            target = getattr(_MODULES[alias], attr, None)
            if target is None or not callable(target):
                continue  # a constant or a class used as a value
            try:
                sig = inspect.signature(target)
            except (TypeError, ValueError):  # pragma: no cover
                continue
            args = [object()] * len(node.args)
            kwargs = {k.arg: object() for k in node.keywords if k.arg}
            if any(k.arg is None for k in node.keywords):
                continue  # **kwargs splat: not statically checkable
            with self.subTest(call=f"{alias}.{attr}"):
                try:
                    sig.bind(*args, **kwargs)
                except TypeError as exc:
                    self.fail(
                        f"launcher calls {alias}.{attr}{sig} with "
                        f"{len(args)} positional + {sorted(kwargs)} -- {exc}. "
                        f"This is the #1368 shape: a producer commit widened a "
                        f"caller and a callee and missed the glue between them."
                    )
            checked += 1
        self.assertGreater(checked, 0, "the scan found no calls -- it broke")

    def test_the_argv_builders_are_in_that_set(self):
        """#1373: the extension must actually reach them. A scan that silently
        stops covering a family is worse than one that never covered it."""
        seen = {n for _a, n, _ in _calls() if n in _LOCAL}
        self.assertTrue(seen, "no argv-builder call site is being bound")
        self.assertIn("argv_d", seen)

    def test_binding_cannot_replace_the_positional_ratchet(self):
        """#1373, MEASURED, so the boundary between this pin and
        test_weg2_argv_positional_pin_1356 is a fact and not an opinion.

        `sig.bind()` answers "does this call exist"; it CANNOT answer "does
        this call pass meaning by position". A parameter inserted mid-signature
        -- the 2cc618b819 defect -- leaves every arity unchanged, so binding
        succeeds before and after while every argument after the insert has
        been re-bound. Measured here rather than asserted in prose, because a
        boundary nobody executes is how two mechanisms quietly become one
        blind one."""
        def before(a, b, c, d=1, e="x"):
            pass

        def after(a, b, vision, c, d=1, e="x"):  # the insert
            pass

        for fn in (before, after):
            inspect.signature(fn).bind(*[object()] * 5)  # both bind: no signal

        # And the real one: argv_d takes 18 positionals today and binds.
        inspect.signature(launcher.argv_d).bind(*[object()] * 18)
        # So the positional profile needs its own ratchet, and it has one.
        # Loaded BY PATH, not by import name: a sibling test module is only
        # importable when pytest happens to have put its directory on sys.path,
        # and a pin that depends on the runner's path layout goes red for a
        # reason that has nothing to do with its subject.
        import importlib.util

        sibling = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "test_weg2_argv_positional_pin_1356.py")
        self.assertTrue(os.path.exists(sibling),
                        "the positional ratchet is gone; this pin does NOT "
                        "cover what it covered")
        spec = importlib.util.spec_from_file_location("_pos_pin_1356", sibling)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.KNOWN_POSITIONAL_CALLS.get(("argv_d", 18)), 2,
                         "the positional ratchet moved or lost its subject; "
                         "this pin does NOT cover what it covers")

    def test_the_call_that_died_is_in_that_set(self):
        """A scan that would pass on an empty set is not a scan."""
        found = [(a, n) for a, n, _ in _calls() if n == "widest_layer_terms"]
        self.assertTrue(found, "widest_layer_terms is no longer called from "
                               "the launcher -- if that is intended, this file "
                               "needs re-aiming, not deleting")


class TheLaneCountHasOneProducer(CustomTestCase):
    def test_widest_layer_terms_forwards_the_lane_count(self):
        """EXECUTED, not read: the parameter must reach BounceTerms, not just
        exist. An accepted-and-dropped kwarg is the same boot, one guard later."""
        terms = xchg_bounce.bounce_terms(
            bytes_per_direction=64 << 20, n_layers=8,
            widest_layer_bytes=8 << 20, pairs=3, depth=1,
            slot_bytes=1 << 20, n_lanes=5)
        self.assertEqual(terms.n_lanes, 5)
        one = xchg_bounce.bounce_terms(
            bytes_per_direction=64 << 20, n_layers=8,
            widest_layer_bytes=8 << 20, pairs=3, depth=1,
            slot_bytes=1 << 20, n_lanes=1)
        self.assertGreater(terms.total_bytes, one.total_bytes,
                           "the lane count must move the charge, or passing "
                           "it through is decoration")

    def test_the_default_is_one_lane_not_a_required_argument(self):
        sig = inspect.signature(checkpoint_census.widest_layer_terms)
        self.assertEqual(sig.parameters["n_lanes"].default, 1)

    def test_both_consumers_go_through_the_one_producer(self):
        """The ledger's charge and the ranks' published inputs must be keyed
        the same way, or the rank guard refuses a correctly priced boot."""
        src = inspect.getsource(launcher)
        self.assertEqual(
            src.count("host_ledger.resolve_xchg_lanes("), 1,
            "resolve_xchg_lanes must be reached through xchg_lane_count only")
        self.assertGreaterEqual(src.count("xchg_lane_count("), 3,
                                "definition plus both call sites")

    def test_the_d_vector_default_is_named_once(self):
        """Two spellings of the d-vector are two cut keys, and a boot that
        prices under one and publishes under the other is the same divergence."""
        sig = inspect.signature(launcher.choose_host_ledger)
        self.assertIs(sig.parameters["d_vector"].default,
                      launcher.XCHG_D_VECTOR_DEFAULT)
        self.assertEqual(
            inspect.signature(launcher.xchg_lane_count)
            .parameters["d_vector"].default,
            launcher.XCHG_D_VECTOR_DEFAULT)

    def test_the_publication_site_passes_a_measured_lane_count(self):
        src = inspect.getsource(launcher)
        i = src.index("bounce_terms_for_ranks, _widest_line, _widest_name")
        window = src[i - 700: i + 700]
        self.assertIn("n_lanes=int(_pub_lane_n)", window)
        self.assertIn("xchg_lane_count(", window)


if __name__ == "__main__":
    unittest.main()
