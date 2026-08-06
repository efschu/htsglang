"""#612: the ledger's communicator-group declaration against the real
construction sites.

The declaration disagreed with ``parallel_state`` in BOTH directions:

  * OVER-declared. ``world`` was stated with the default ``use_pynccl=True``
    while ``init_world_group`` passes ``use_pynccl=False`` unconditionally, so
    every ``nccl_signature`` this launch produced carried a member that
    allocates nothing. A phantom member cannot under-charge, but it invalidates
    measurements that are still exactly right, and it can turn a launch that
    builds no communicator into an UNBOUNDED refusal nobody can satisfy by
    measuring.
  * UNDER-declared. ``pdmux_prefill_tp``, ``dcp_spill``, ``attn_cp``,
    ``attention_tp``, ``moe_dp``, ``moe_ep`` and ``moe_tp`` were absent. Two of
    them are SECOND communicators over ranks another group already covers, so
    their buffers were charged to nobody and did not move the signature either.

The pin below is the #504-a one-source-of-truth pattern: the runtime group
names are read out of ``parallel_state`` by AST at test time, so a group added
upstream fails here instead of quietly leaving the ledger a group short. It is
deliberately a STATIC read and not a re-implementation of the conditions -- a
test that restated the conditions would keep passing after the real ones
changed, which is the failure this file exists to prevent.
"""

import ast
import os
import types
import unittest
from typing import Dict, Optional

from sglang.srt.mem_ledger.engine import (
    RUNTIME_COMMUNICATOR_GROUPS,
    communicator_groups_from_server_args,
)
from sglang.srt.mem_ledger.nccl_transport import (
    CommunicatorGroup,
    classify_communicator_groups,
    nccl_signature,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _parallel_state_path() -> str:
    import sglang.srt.distributed.parallel_state as ps

    return ps.__file__


def _constructed_groups() -> Dict[str, Optional[bool]]:
    """``{group_name: use_pynccl literal or None}`` for every GroupCoordinator
    construction in ``parallel_state``.

    ``None`` means the call site does not pass the flag, i.e. it takes
    ``init_model_parallel_group``'s default.
    """
    tree = ast.parse(open(_parallel_state_path()).read())
    out: Dict[str, Optional[bool]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        use_pynccl: Optional[bool] = None
        for kw in node.keywords:
            if kw.arg == "group_name" and isinstance(kw.value, ast.Constant):
                name = kw.value.value
            if kw.arg == "use_pynccl" and isinstance(kw.value, ast.Constant):
                use_pynccl = kw.value.value
        if name is not None:
            out[str(name)] = use_pynccl
    return out


class TestTheDeclarationNamesEveryGroupTheRuntimeBuilds(unittest.TestCase):
    def test_no_runtime_group_is_missing_from_the_declaration(self):
        built = set(_constructed_groups())
        declared = set(RUNTIME_COMMUNICATOR_GROUPS)
        missing = sorted(built - declared)
        self.assertEqual(
            missing,
            [],
            f"parallel_state constructs {missing} and the ledger's "
            "RUNTIME_COMMUNICATOR_GROUPS does not mention them. An undeclared "
            "group allocates communicator buffers that no ledger term charges "
            "and that do not move nccl_signature, so a measurement taken "
            "without it is silently reused for a launch that has it",
        )

    def test_no_declared_group_is_absent_from_the_runtime(self):
        built = set(_constructed_groups())
        declared = set(RUNTIME_COMMUNICATOR_GROUPS)
        phantom = sorted(declared - built)
        self.assertEqual(
            phantom,
            [],
            f"the ledger declares {phantom}, which parallel_state never "
            "constructs. A phantom group poisons nccl_signature and can refuse "
            "a launch that in fact builds nothing",
        )

    def test_every_group_the_declaration_emits_is_an_inventoried_name(self):
        sa = types.SimpleNamespace(
            tp_size=4,
            pp_size=2,
            dcp_size=2,
            ep_size=2,
            moe_dp_size=2,
            attn_cp_size=2,
            dp_size=2,
            enable_dp_attention=True,
            enable_pdmux=True,
            enable_symm_mem=True,
        )
        old = os.environ.get("SGLANG_KVSO_DECOUPLE")
        os.environ["SGLANG_KVSO_DECOUPLE"] = "1"
        try:
            names = [
                g.name for g in communicator_groups_from_server_args(sa, list(range(8)))
            ]
        finally:
            if old is None:
                os.environ.pop("SGLANG_KVSO_DECOUPLE", None)
            else:
                os.environ["SGLANG_KVSO_DECOUPLE"] = old
        self.assertEqual(len(names), len(set(names)), names)
        for name in names:
            self.assertIn(name, RUNTIME_COMMUNICATOR_GROUPS, name)


class TestTheWorldGroupTransportIsQuotedNotAssumed(unittest.TestCase):
    def test_init_world_group_still_passes_use_pynccl_false(self):
        self.assertIs(
            _constructed_groups()["world"],
            False,
            "init_world_group no longer passes use_pynccl=False. The ledger "
            "declares the world group on that literal; if the construction "
            "site changed, the declaration has to change with it",
        )

    def test_the_declared_world_group_carries_that_flag(self):
        sa = types.SimpleNamespace(tp_size=2, pp_size=1, dcp_size=1)
        groups = {g.name: g for g in communicator_groups_from_server_args(sa, [0, 1])}
        self.assertFalse(groups["world"].use_pynccl)
        self.assertEqual(groups["world"].world_size, 2)

    def test_the_world_group_no_longer_enters_the_signature(self):
        """The falsifier for the over-declaration.

        A launch's measured NCCL figure must be filed against the groups that
        ALLOCATE. With the world group declared as a PyNccl group, the digest
        moved whenever the world size moved, even though nothing was allocated
        for it.
        """
        sa = types.SimpleNamespace(tp_size=2, pp_size=1, dcp_size=1)
        groups = communicator_groups_from_server_args(sa, [0, 1])
        tp_only = tuple(g for g in groups if g.name == "tp")
        self.assertEqual(nccl_signature(groups), nccl_signature(tp_only))

        wrong = tuple(
            (
                CommunicatorGroup(name=g.name, world_size=g.world_size, use_pynccl=True)
                if g.name == "world"
                else g
            )
            for g in groups
        )
        self.assertNotEqual(
            nccl_signature(wrong),
            nccl_signature(groups),
            "the pre-#612 declaration and the corrected one must not digest "
            "alike, or this test proves nothing",
        )

    def test_the_composite_verdict_is_unchanged_for_a_real_launch(self):
        """The world fix may not flip a launch to NOT_APPLICABLE.

        Whenever the world group spans more than one rank, so does tp or pp,
        and both are declared with the default transport. This is the reason
        the correction is safe in the under-charge direction.
        """
        sa = types.SimpleNamespace(tp_size=2, pp_size=1, dcp_size=1)
        groups = communicator_groups_from_server_args(sa, [0, 1])
        verdicts = classify_communicator_groups(groups)
        self.assertTrue(any(v.builds_nccl for v in verdicts), verdicts)


class TestTheUnderDeclaredGroupsAppear(unittest.TestCase):
    BASE = dict(tp_size=2, pp_size=1, dcp_size=1)

    def _names(self, **over):
        sa = types.SimpleNamespace(**{**self.BASE, **over})
        return [g.name for g in communicator_groups_from_server_args(sa, [0, 1])]

    def test_the_pdmux_duplicate_tp_group_is_declared(self):
        self.assertNotIn("pdmux_prefill_tp", self._names())
        names = self._names(enable_pdmux=True)
        self.assertIn("pdmux_prefill_tp", names)

    def test_the_pdmux_group_changes_what_a_measurement_is_valid_for(self):
        without = communicator_groups_from_server_args(
            types.SimpleNamespace(**self.BASE), [0, 1]
        )
        with_pdmux = communicator_groups_from_server_args(
            types.SimpleNamespace(**{**self.BASE, "enable_pdmux": True}), [0, 1]
        )
        self.assertNotEqual(
            nccl_signature(without),
            nccl_signature(with_pdmux),
            "a second communicator over the same tp ranks doubles the buffer "
            "set; a measurement taken without it may not be reused with it",
        )

    def test_the_kv_session_offload_spill_group_is_declared_with_its_env(self):
        old = os.environ.get("SGLANG_KVSO_DECOUPLE")
        try:
            os.environ["SGLANG_KVSO_DECOUPLE"] = "1"
            self.assertIn("dcp_spill", self._names(tp_size=2, dcp_size=2))
            # Only with DCP on: the spill communicator is built inside the DCP
            # branch, so it cannot exist without it.
            self.assertNotIn("dcp_spill", self._names(dcp_size=1))
            os.environ["SGLANG_KVSO_DECOUPLE"] = "0"
            self.assertNotIn("dcp_spill", self._names(tp_size=2, dcp_size=2))
        finally:
            if old is None:
                os.environ.pop("SGLANG_KVSO_DECOUPLE", None)
            else:
                os.environ["SGLANG_KVSO_DECOUPLE"] = old

    def test_the_moe_groups_are_declared_with_their_real_transport(self):
        sa = types.SimpleNamespace(**{**self.BASE, "ep_size": 1})
        groups = {g.name: g for g in communicator_groups_from_server_args(sa, [0, 1])}
        self.assertIn("moe_ep", groups)
        self.assertFalse(
            groups["moe_ep"].use_pynccl,
            "parallel_state constructs moe_ep with use_pynccl=False",
        )

    def test_a_group_that_aliases_tp_is_not_declared(self):
        """A sub-group whose size equals tp_size ALIASES _TP.

        ``_ATTN_CP = _TP`` / ``_MOE_EP = _TP`` / ``_MOE_DP = _TP`` build no
        communicator of their own, so declaring them would be the phantom
        member again -- one rank set counted twice.
        """
        sa = types.SimpleNamespace(
            **{**self.BASE, "attn_cp_size": 2, "ep_size": 2, "moe_dp_size": 2}
        )
        names = [g.name for g in communicator_groups_from_server_args(sa, [0, 1])]
        self.assertNotIn("attn_cp", names)
        self.assertNotIn("moe_ep", names)
        self.assertNotIn("moe_dp", names)

    def test_a_sub_group_that_differs_from_tp_is_declared(self):
        """The same config still builds attention_tp, and it is now stated.

        With attn_cp_size == tp_size the attention-TP width is
        tp // attn_cp // attn_dp = 1, which differs from tp_size, so
        parallel_state takes the else branch and constructs a real (single-rank)
        group. Before #612 the ledger did not know it existed.
        """
        sa = types.SimpleNamespace(
            **{**self.BASE, "attn_cp_size": 2, "ep_size": 2, "moe_dp_size": 2}
        )
        groups = {g.name: g for g in communicator_groups_from_server_args(sa, [0, 1])}
        self.assertIn("attention_tp", groups)
        self.assertEqual(groups["attention_tp"].world_size, 1)


class TestTheDeclarationCitesItsSites(unittest.TestCase):
    def test_every_inventory_entry_names_a_construction_site(self):
        for name, rule in RUNTIME_COMMUNICATOR_GROUPS.items():
            self.assertIn(
                "parallel_state.py:",
                rule,
                f"{name} states no construction site; a declaration a reader "
                "cannot check against the code is the drift this pin exists "
                "to catch",
            )


if __name__ == "__main__":
    unittest.main()
