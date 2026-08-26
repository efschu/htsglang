"""#881b: predicates that NAME A FLAG where they READ PROCESS STATE.

The generic form of the error that struck five times in one day, most recently
inside the CORRECTION of an earlier instance of itself. A function reads an
installable process global -- ``get_X()``, paired with a ``set_X()`` that more
than one call site invokes -- but its docstring names ONE ``--flag`` as the
input. A reader then answers the reachability question by checking that flag,
which is not the input, and concludes the opposite of the truth.

THE MEASURED SPECIMEN (#881): ``uneven_dcp_kv_replicated`` documented
``--rank-tp-ratio``. On this rig that flag is None while the predicate is TRUE,
because ``phase_flip_boot`` installs the plan from ``--phase-flip-tp-vector`` --
a different flag on a different axis -- before the TP worker is built. The
consequence is a KV pool ROW STRIDE (#345), i.e. silent corruption, not a
sizing error.

WHY THIS IS DERIVED AND NOT A LIST. The criterion is structural and computed
here every run:

    a process-state accessor ``get_X`` whose installer ``set_X`` has >= 2
    NON-DEFINITION call sites, read by a function whose docstring names a
    ``--flag``

Nothing about the known instance is privileged. Measured today: 31 such
accessor pairs exist, and ``get_tp_partition_ratios`` alone has FOUR installers
-- one more than the two the specimen investigation found by hand.

WHAT IT IS NOT. This does not claim the documented flag is WRONG; usually it is
the common installer. It claims the docstring presents a flag as THE input for
state that has several, which is what makes "I checked the flag" sound like an
answer. The remedy is wording, not behaviour -- see #881 for the shape.

WHERE IT STRUCTURALLY DOES NOT REACH, named rather than closed:
  * a docstring that names NOTHING -- no flag to catch, and a reader with no
    wrong answer handed to them, so it is out of scope by intent;
  * a flag and its process state that share a NAME, where reading the flag
    happens to be right and the scan cannot tell the two apart;
  * installers reached through a wrapper or a registry, where ``set_X(`` never
    appears literally -- the installer count is then a LOWER BOUND, so this
    scan under-reports and never over-reports on that axis;
  * dynamic docstrings (``__doc__`` assigned at runtime).

Hermetic: source scanning only. No CUDA, no boot, no card.
"""

import collections
import pathlib
import re
import unittest

from sglang.test.test_utils import CustomTestCase

_SET = re.compile(r"\b(set_[a-z_]{4,})\(")
_DEF = re.compile(r"def (set_[a-z_]{4,})\(")
#: 6000, not 900: the first version capped the docstring window at 900 chars
#: and then FAILED TO SEE ITS OWN SPECIMEN, because #881's corrected
#: docstring is longer than that. A scan whose window is shorter than the
#: documentation it reads reports a false GREEN -- the one direction that is
#: indistinguishable from "nothing to find".
_DOCFN = re.compile(r'def ([a-z_]+)\([^)]*\)[^:]*:\s*\n\s*"""(.{0,6000}?)"""', re.S)
_FLAG = re.compile(r"--[a-z][a-z0-9-]{3,}")

#: Sites the scan finds, each with a decided verdict. NOT the criterion -- the
#: criterion is derived above. This is the RECORD that each finding was looked
#: at, so a NEW one fails the gate instead of joining a backlog.
RECORDED = {
    (
        "get_tp_partition_ratios",
        "uneven_dcp_kv_replicated",
    ): "#881 FIXED -- the specimen",
    ("get_tp_partition_ratios", "_uneven_tp_num_kv_heads"): (
        "THE MOST LOAD-BEARING SIBLING: it computes the kv-head count, on the "
        "very axis that carried two wrong judgments. Documented as "
        "--rank-tp-ratio while reading the process plan. Wording only; the "
        "behaviour reads the plan and is correct."
    ),
    # The remaining reads of the same accessor. All verified to READ the plan
    # (behaviour correct); all document a single --flag for state with four
    # installers. Wording class, one decision: they are recorded as siblings of
    # the specimen rather than fixed one at a time, because rewriting eleven
    # docstrings in a pass nobody asked for is how a sweep becomes a rewrite.
    # #881's wording is the template when any of them is next touched.
    ("get_tp_partition_ratios", "_reject_uneven_tp_mla"): "sibling, wording",
    (
        "get_tp_partition_ratios",
        "_reject_uneven_tp_unaware_attention",
    ): "sibling, wording",
    ("get_tp_partition_ratios", "_replication_axis_lines"): "sibling, wording",
    ("get_tp_partition_ratios", "_resolved_weight_vector"): "sibling, wording",
    ("get_tp_partition_ratios", "assert_pd_decode_dcp_supported"): "sibling, wording",
    ("get_tp_partition_ratios", "installed_family_ratios"): "sibling, wording",
    (
        "get_tp_partition_ratios",
        "resident_fraction_held_at_base_plan",
    ): "sibling, wording",
    ("get_tp_partition_ratios", "sync_fixed_hicache_size"): "sibling, wording",
    ("get_tp_partition_ratios", "tp_loaded_shard_start"): "sibling, wording",
    ("get_tp_partition_ratios", "tp_vocab_ratios"): "sibling, wording",
    (
        "get_tp_partition_ratios",
        "validate_pd_dcp_token_shard_contract",
    ): "sibling, wording",
    ("get_cp_token_ratios", "build_kv_reshard_runtime"): "token axis, same shape",
    ("get_cp_token_ratios", "effective_flip_token_vector"): "token axis, same shape",
    ("get_context", "_from_flag"): "moe resident fraction; the NAME says flag",
}
#: ``get_server_args`` is EXCLUDED by rule, not by taste: it returns the whole
#: parsed config, so a docstring naming a flag it contains is ACCURATE. The
#: class is about state whose installers diverge from the documented flag.
EXCLUDED_ACCESSORS = {"get_server_args"}


def _files() -> dict:
    import sglang.srt as srt

    root = pathlib.Path(list(srt.__path__)[0])
    return {
        str(p).split("srt/")[-1]: p.read_text(errors="ignore")
        for p in root.rglob("*.py")
    }


def _findings() -> dict:
    files = _files()
    blob = "\n".join(files.values())
    calls = collections.Counter(_SET.findall(blob))
    defs = collections.Counter(_DEF.findall(blob))
    multi = {n: calls[n] - defs[n] for n in calls if calls[n] - defs[n] >= 2}
    paired = {
        n: c for n, c in multi.items() if f"def {n.replace('set_', 'get_')}(" in blob
    }
    out = {}
    for setter, count in paired.items():
        getter = setter.replace("set_", "get_")
        if getter in EXCLUDED_ACCESSORS:
            continue
        for fname, text in files.items():
            if getter + "(" not in text:
                continue
            for m in _DOCFN.finditer(text):
                body = text[m.end() : m.end() + 1200]
                if getter + "(" in body and _FLAG.search(m.group(2)):
                    out[(getter, m.group(1))] = (fname, count)
    return out


class TestFlagNamedProcessState(CustomTestCase):
    def test_the_criterion_finds_more_installers_than_hand_search_did(self):
        """Derived, not inherited: the specimen hunt found two installers of
        the plan by hand. The structural count must not be smaller."""
        files = _files()
        blob = "\n".join(files.values())
        n = len(_SET.findall(blob)) - len(_DEF.findall(blob))
        self.assertGreater(n, 20, "the installer scan collapsed")
        calls = collections.Counter(_SET.findall(blob))
        defs = collections.Counter(_DEF.findall(blob))
        self.assertGreaterEqual(
            calls["set_tp_partition_ratios"] - defs["set_tp_partition_ratios"],
            3,
            "get_tp_partition_ratios should have at least three installers; a "
            "smaller count means the scan stopped seeing them, not that they "
            "went away",
        )

    def test_it_covers_the_specimen(self):
        self.assertIn(
            ("get_tp_partition_ratios", "uneven_dcp_kv_replicated"), _findings()
        )

    def test_it_finds_siblings_on_the_same_axis_that_never_hurt(self):
        """Generality: `_uneven_tp_num_kv_heads` computes the kv-head count on
        the very axis that produced two wrong judgments, and no instance fix
        ever touched its wording."""
        found = _findings()
        self.assertIn(("get_tp_partition_ratios", "_uneven_tp_num_kv_heads"), found)
        self.assertGreaterEqual(
            len({k for k in found if k[0] == "get_tp_partition_ratios"}),
            4,
            "the scan no longer sees the sibling cluster on this accessor",
        )

    def test_every_finding_has_a_recorded_verdict(self):
        """THE GATE. A new flag-named process-state read fails here."""
        found = _findings()
        unrecorded = sorted(k for k in found if k not in RECORDED)
        self.assertEqual(
            unrecorded,
            [],
            "functions document a --flag while reading process state that has "
            "several installers, and nobody has decided what that means: "
            + ", ".join(f"{g}<-{fn} ({found[(g, fn)][0]})" for g, fn in unrecorded),
        )

    def test_whole_config_accessors_are_excluded_by_rule(self):
        """No crying wolf: `get_server_args` returns the parsed config, so a
        docstring naming a flag it carries is ACCURATE, not misleading."""
        self.assertIn("get_server_args", EXCLUDED_ACCESSORS)
        self.assertEqual(
            [k for k in _findings() if k[0] == "get_server_args"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
