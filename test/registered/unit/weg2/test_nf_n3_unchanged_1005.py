"""NF N = 3 UNCHANGED 1005 (Auftrag 1518, nf-offen-02): the hardware-generic
building blocks that the NF release candidate (3bfee09511) adds to the boot
path leave the REFERENCE configuration (NVML order 3080-20G, 5090, 3080-20G =
N = 3) exactly as it was.

WHY: the candidate puts five calls into ``launcher.main`` (launcher.py, right
after ``order_cards(resolve_cards())``)::

    log(topology_check_line(ns, cards))
    log(inventory_check_line(ns, cards))
    refusals.flush(log)
    apply_inventory_derivation(ns, cards, log)
    apply_bar1_windows(ns, cards, log)
    configure_xchg_region(len(cards), log)

and claims "no-op at N = 3". The 27B line carries ``test_dual_fixes_flip_unchanged_1003``
for the Dual fixes; it does not cover these blocks, and the NF tree had no
"unchanged" test at all (done/nf-release-kandidat-bericht-1005.md section 4 last
bullet, done/1489b-bericht.md section 3). This file runs the SAME five calls in
main's order, on the NF release profile argv (``hw_sim.MODELS["NF"]``: weight
source exchange, the 3-entry vectors of nf.env) and the reference inventory
(``hw_sim.REFERENCE_RIG`` through the NVML replay seam, ``launcher.resolve_cards``
+ ``order_cards`` -- the launcher's own card producer), and asserts:

  (a) ``topology_check_line`` returns the normal ``HW-TOPOLOGY N=3 ...`` line,
      does not raise, remembers no refusal;
  (b) ``inventory_check_line`` is ``... MATCH`` (not DERIVED, not MISMATCH);
  (c) the three apply/configure functions change neither ``vars(ns)`` nor the
      environment nor the exchange-region geometry nor the active inventory view,
      and log nothing; the argv builders give the same tokens before and after;
  (d) ``refusals.forced_list()`` stays empty -- also in a ``--force`` boot, i.e.
      there is nothing a forced boot would have to pass at N = 3, and
      ``refusals.flush`` prints nothing;
  (e) the hw_sim cell of the reference rig equals the golden below, and the
      N = 3 cells of ``grid_golden_1to8.json``.

GOLDEN (fixtures/nf_n3_unchanged_1005/nf_n3_reference_3bfee09511.json): NOT
invented -- generated from the tree 3bfee09511 itself (the NF release candidate,
detached worktree, CPU only, no GPU, no NVML device read: replay seam)::

    CUDA_VISIBLE_DEVICES= PYTHONPATH=python:test/registered/unit/weg2 \\
        python3 test/registered/unit/weg2/test_nf_n3_unchanged_1005.py \\
        --write-golden test/registered/unit/weg2/fixtures/nf_n3_unchanged_1005/nf_n3_reference_3bfee09511.json

REGENERATE it only after a change that moves an N = 3 value ON PURPOSE (a new
reference vector, a new topology line): the diff of the golden is then the review
object. A red test here without such a change means the reference configuration
moved by accident.

WHAT THIS DOES NOT PROVE (HOCHRECHNUNG != MESSUNG): runtime behaviour on the
metal -- allocator residue, the vectors as the ranks really read them, the
argv the ranks really get, the Planner card at boot. That is the smoke boot
(``deskq/bl/vergleich_argv_vektoren_1005.sh`` compares two boot logs).

GPU-free, NVML-free.
"""

import copy
import json
import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import hw_sim as HS
from sglang.srt.weg2 import inventory_view as IV
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import profile_records as PR
from sglang.srt.weg2 import refusals
from sglang.srt.weg2 import weight_exchange_region as XR
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-weg2-unit")

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "fixtures", "nf_n3_unchanged_1005", "nf_n3_reference_3bfee09511.json")
GRID_GOLDEN = os.path.join(HERE, "fixtures", "hw_sim_1003", "grid_golden_1to8.json")
REAL_PROFILES = os.path.join(os.environ.get("HW_GENERIC_DOCKER_DIR", "/spinning/gpu-arb/docker"),
                             "profiles_release")

#: the launcher flag dests the boot reads as vectors / windows (all of them the
#: hardware-generic blocks may touch)
_NS_DESTS = tuple(L.POSITIONAL_VECTOR_FLAGS) + ("extra_p", "extra_d", "env_p", "env_d", "dual_layout",
                                                  "dual_share", "d_barlink_bar1_window_mib", "force", "cards")


def reference_cards():
    """The reference inventory as the boot gets it: NVML replay of REFERENCE_RIG
    through the launcher's own producer, then main's ``order_cards(resolve_cards())``."""
    with HS.replayed(HS.REFERENCE_RIG):
        return L.order_cards(L.resolve_cards())


def nf_ns(model):
    return L.build_parser().parse_args(
        ["--tree", "/sim", "--tag", "sim", "--profile", model.profile,
         "--model", F.profile_row(model.profile).formats[model.weight_format].checkpoint, *model.argv])


def jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: list(o) if isinstance(o, (tuple, set)) else repr(o)))


def xchg_geometry():
    return {name: jsonable(getattr(XR, name)) if name != "MATRIX_PAYLOAD_STRUCT" else XR.MATRIX_PAYLOAD_STRUCT.format
            for name in XR._GEOMETRY_NAMES}


def argv_tokens(ns):
    """The P and D argv the launcher's builders give for three sentinel budgets
    with the BAR1 windows the NAMESPACE carries (what main passes on)."""
    pk = {"window_mib": str(ns.p_barlink_bar1_window_mib)}
    dk = {"window_mib": str(ns.d_barlink_bar1_window_mib)}
    p = L.argv_p("py", L.MODEL_DEFAULT, [1, 1, 1], 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], p_bs=1, **pk)
    d = L.argv_d("py", L.MODEL_DEFAULT, [1, 1, 1], 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], d_bs=1, **dk)
    return {"p": list(p), "d": list(d)}


def record_values(profile):
    """Every record of the profile as the launch reads it (``form.profile_constant``,
    which applies an installed inventory view) next to the raw file value."""
    out = {}
    for r in PR.records(profile):
        try:
            read = jsonable(F.profile_constant(r.name, profile))
        except KeyError:        # a record the profile row does not expose as a constant (borrowed rows)
            read = jsonable(r.value)
        out[r.name] = {"file": jsonable(r.value), "read": read}
    return out


def image(ns, env):
    """Everything the five calls could move, as JSON-comparable data."""
    return {
        "ns": jsonable({k: getattr(ns, k, None) for k in _NS_DESTS}),
        "ns_all": jsonable(vars(ns)),
        "env": dict(env),
        "environ": dict(os.environ),
        "vector_lengths": L.positional_vector_lengths(ns),
        "xchg": xchg_geometry(),
        "argv": argv_tokens(ns),
        "records": record_values(ns.profile),
        "view_active": jsonable(IV.active()),
        "forced": refusals.forced_list(),
    }


def boot_sequence(ns, cards, env, log):
    """launcher.main's hardware block, in main's order (launcher.py right after
    ``cards = order_cards(resolve_cards())``). Returns the topology and inventory
    lines main would log."""
    topo = L.topology_check_line(ns, cards, env)
    log(topo)
    inv = L.inventory_check_line(ns, cards, env)
    log(inv)
    refusals.flush(log)
    L.apply_inventory_derivation(ns, cards, log, env)
    L.apply_bar1_windows(ns, cards, log)
    L.configure_xchg_region(len(cards), log)
    return topo, inv


def golden_of_tree():
    """The comparable core, computed from the running tree (the --write-golden source)."""
    model = HS.MODELS["NF"]
    cards = reference_cards()
    ns, env, lines = nf_ns(model), dict(model.env), []
    topo, inv = boot_sequence(ns, cards, env, lines.append)
    cell = HS.simulate("ref", list(HS.REFERENCE_RIG), model)
    return {
        "_how": "generated by test_nf_n3_unchanged_1005.py --write-golden from the tree 3bfee09511 "
                "(detached worktree, CPU only, NVML replay seam); see the module docstring",
        "model": model.key,
        "inventory": [f"{c.name}/{c.total_mib}MiB/sm{c.cc[0]}{c.cc[1]}/bar1={c.bar1_total_mib}" for c in cards],
        "topology_line": topo,
        "inventory_line": inv,
        "boot_log_lines": lines,
        "vector_lengths": L.positional_vector_lengths(ns),
        "vectors": jsonable({k: getattr(ns, k) for k in L._TOPOLOGY_VECTOR_FLAGS
                             if getattr(ns, k, None) not in (None, "")}),
        "windows": {"p": str(ns.p_barlink_bar1_window_mib), "d": str(ns.d_barlink_bar1_window_mib)},
        "xchg": xchg_geometry(),
        "records_inventory": list(L.records_inventory(model.profile)[0]),
        "hw_sim_cell": {"result": cell.result, "code": cell.code, "blockers": list(cell.blockers),
                        "order": list(cell.order), "argv": cell.argv, "notes": list(cell.notes)},
    }


def load_golden():
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


class NfReferenceN3Unchanged(unittest.TestCase):
    """The NF release profile on the reference inventory through main's hardware block."""

    FORCED = False

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        refusals.arm(False)
        self.addCleanup(refusals.arm, False)
        IV.clear_active()
        self.addCleanup(IV.clear_active)
        # the process-global region geometry is whatever an earlier test left: the
        # baseline of this test is the reference geometry; restore afterwards
        prev = XR.configure(XR.DEFAULT_N_CARDS)
        self.addCleanup(XR.configure, prev)
        self.model = HS.MODELS["NF"]
        self.cards = reference_cards()
        self.env = dict(self.model.env)
        self.ns = nf_ns(self.model)
        if self.FORCED:
            refusals.arm(True)      # exactly what main does for --force (launcher.py: refusals.arm(True, sink=...))
        self.lines = []

    # (a)
    def test_a_topology_line_is_the_normal_one_and_remembers_nothing(self):
        line = L.topology_check_line(self.ns, self.cards, self.env)
        self.assertEqual(line, load_golden()["topology_line"])
        self.assertTrue(line.startswith("HW-TOPOLOGY N=3: P = TP1 x PP3, D = TP3 x PP1, rank-gpu-id 0,1,2"), line)
        self.assertIn("proven on metal (N in [3])", line)
        self.assertEqual(refusals.forced_list(), [])

    # (b)
    def test_b_inventory_check_is_match_not_derived_not_mismatch(self):
        line = L.inventory_check_line(self.ns, self.cards, self.env)
        self.assertEqual(line, load_golden()["inventory_line"])
        self.assertTrue(line.endswith(" MATCH"), line)
        self.assertNotIn("DERIVED", line)
        self.assertNotIn("MISMATCH", line)
        self.assertEqual(refusals.forced_list(), [])

    # (c)
    def test_c_apply_and_configure_change_nothing_and_log_nothing(self):
        before = image(self.ns, self.env)
        notes = L.apply_inventory_derivation(self.ns, self.cards, self.lines.append, self.env)
        self.assertEqual(notes, [])
        self.assertIsNone(L.apply_bar1_windows(self.ns, self.cards, self.lines.append))
        L.configure_xchg_region(len(self.cards), self.lines.append)
        self.assertEqual(self.lines, [], "the N = 3 no-op paths must not log (no HW-DERIVE / BAR1-WINDOW / "
                                        "WEG2-XCHG-REGION line)")
        after = image(self.ns, self.env)
        for key in before:
            self.assertEqual(before[key], after[key], f"{key} moved at N = 3")
        self.assertIsNone(IV.active())

    # (c) + (d): the whole block in main's order
    def test_cd_the_whole_hardware_block_is_a_no_op_on_the_reference_rig(self):
        before = image(self.ns, self.env)
        topo, inv = boot_sequence(self.ns, self.cards, self.env, self.lines.append)
        after = image(self.ns, self.env)
        self.assertEqual(self.lines, [topo, inv], "main logs exactly the two check lines at N = 3")
        for key in before:
            self.assertEqual(before[key], after[key], f"{key} moved at N = 3")
        for marker in ("HW-DERIVE", "BAR1-WINDOW", "WEG2-XCHG-REGION", "FORCED", "MISMATCH", "DERIVED"):
            self.assertFalse(any(marker in ln for ln in self.lines), marker)
        self.assertEqual(refusals.forced_list(), [])
        self.assertEqual(refusals.flush(self.lines.append), 0)
        # the records are read as the file has them
        for name, rec in after["records"].items():
            self.assertEqual(rec["file"], rec["read"], name)

    # (d)
    def test_d_nothing_to_force_past_at_n3(self):
        refusals.arm(True)     # a --force boot: any value refusal at N = 3 would be remembered here
        boot_sequence(self.ns, self.cards, self.env, self.lines.append)
        self.assertEqual(refusals.forced_list(), [], "a --force boot has nothing to go past on the reference rig")
        self.assertEqual(self.lines, [load_golden()["topology_line"], load_golden()["inventory_line"]])

    # (e)
    def test_e_vectors_windows_geometry_equal_the_golden(self):
        g = load_golden()
        boot_sequence(self.ns, self.cards, self.env, self.lines.append)
        self.assertEqual(self.lines, g["boot_log_lines"])
        self.assertEqual(L.positional_vector_lengths(self.ns), g["vector_lengths"])
        self.assertEqual(set(g["vector_lengths"].values()), {3}, "every NF vector has one entry per card")
        vectors = jsonable({k: getattr(self.ns, k) for k in L._TOPOLOGY_VECTOR_FLAGS
                            if getattr(self.ns, k, None) not in (None, "")})
        self.assertEqual(vectors, g["vectors"])
        self.assertEqual({"p": str(self.ns.p_barlink_bar1_window_mib), "d": str(self.ns.d_barlink_bar1_window_mib)},
                         g["windows"])
        self.assertEqual((g["windows"]["p"], g["windows"]["d"]),
                         (L.P_BARLINK_BAR1_WINDOW_MIB, L.D_BARLINK_BAR1_WINDOW_MIB))
        self.assertEqual(xchg_geometry(), g["xchg"])
        self.assertEqual(list(L.records_inventory(self.model.profile)[0]), g["records_inventory"])
        self.assertEqual(g["inventory"], [f"{c.name}/{c.total_mib}MiB/sm{c.cc[0]}{c.cc[1]}/bar1={c.bar1_total_mib}"
                                          for c in self.cards])

    def test_e_the_hw_sim_cell_equals_the_golden_and_the_grid(self):
        g = load_golden()["hw_sim_cell"]
        cell = HS.simulate("ref", list(HS.REFERENCE_RIG), self.model)
        self.assertEqual({"result": cell.result, "code": cell.code, "blockers": list(cell.blockers),
                          "order": list(cell.order), "argv": cell.argv, "notes": list(cell.notes)}, g)
        self.assertEqual((cell.result, cell.blockers), (HS.RUNS, []))
        with open(GRID_GOLDEN, encoding="utf-8") as fh:
            grid = json.load(fh)
        self.assertEqual(grid["3x ref 5090+3080 | NF"], [HS.RUNS, "", []])
        # the simulation leaves no inventory view installed
        self.assertIsNone(IV.active())


class NfReferenceN3UnchangedForced(NfReferenceN3Unchanged):
    """The same assertions in a --force boot (refusals armed before the block)."""

    FORCED = True


@unittest.skipUnless(os.path.isfile(os.path.join(REAL_PROFILES, "nf.env")),
                     "docker/profiles_release/nf.env is not on this box")
class NfRealProfileArgsN3(unittest.TestCase):
    """The REAL NF release profile (PROFILE_ARGS of nf.env, sourced the way the
    entrypoint does) instead of the embedded copy: the same block, same verdict."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        refusals.arm(False)
        self.addCleanup(refusals.arm, False)
        IV.clear_active()
        self.addCleanup(IV.clear_active)
        prev = XR.configure(XR.DEFAULT_N_CARDS)
        self.addCleanup(XR.configure, prev)

    def test_the_real_nf_env_runs_through_the_block_unchanged(self):
        table = HS.models_from_profiles(REAL_PROFILES)
        model = table["NF"]
        self.assertNotEqual(model.argv, HS.MODELS["NF"].argv, "the real file must have been read")
        ns, env, lines = nf_ns(model), dict(model.env), []
        cards = reference_cards()
        before = image(ns, env)
        topo, inv = boot_sequence(ns, cards, env, lines.append)
        after = image(ns, env)
        self.assertEqual(lines, [topo, inv])
        self.assertEqual(topo, load_golden()["topology_line"])
        self.assertTrue(inv.endswith(" MATCH"), inv)
        for key in before:
            self.assertEqual(before[key], after[key], key)
        self.assertEqual(set(L.positional_vector_lengths(ns).values()), {3})
        cell = HS.simulate("ref", list(HS.REFERENCE_RIG), model)
        self.assertEqual((cell.result, cell.blockers), (HS.RUNS, []))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-golden":
        os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
        with open(sys.argv[2], "w", encoding="utf-8") as fh:
            json.dump(golden_of_tree(), fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        print("golden written:", sys.argv[2])
    else:
        unittest.main()
