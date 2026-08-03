# SPDX-License-Identifier: Apache-2.0
"""#518: an op with no tensor argument cannot be registered device-only.

``ggml_moe_get_block_size(int type) -> int`` was registered for the CUDA
dispatch key alone. The dispatcher picks a backend from the tensor arguments,
and there are none, so the call could not be routed at all: every invocation
raised ``NotImplementedError: There were no tensor arguments to this function
... but no fallback function is registered``. Not arch-specific and not
data-dependent -- it failed identically on sm86 and sm120, and the raise
happens in the dispatcher BEFORE any kernel, which is what makes it testable
here without a GPU or a wheel.

Two levels, both hermetic:

* :class:`TestDispatcherMechanism` reproduces the defect and the fix on
  throwaway ops defined in this process. This is the behaviour claim, executed.
* :class:`TestRegistrationRatchet` reads the shipped ``TORCH_LIBRARY_FRAGMENT``
  sources and requires that NO op whose schema has no ``Tensor`` argument is
  bound to a device key -- so the next probe added to the extension cannot
  reintroduce the class.

The #81 family fix pattern (a keyless ``m.impl``, which registers catch-all)
is what the extension already uses for ``apply_token_bitmask_inplace_cuda``
and the ``es_*`` grouped-mm ops.

Usage:
    python3 -m pytest test/registered/unit/quantization/test_no_tensor_op_dispatch_518.py -v
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_EXTENSIONS = (
    _ROOT / "sgl-kernel" / "csrc" / "common_extension.cc",
    _ROOT / "sgl-kernel" / "csrc" / "common_extension_musa.cc",
    _ROOT / "sgl-kernel" / "csrc" / "common_extension_rocm.cc",
)

#: Dispatch keys that name a device. An op with no tensor argument bound to
#: one of these is unreachable.
_DEVICE_KEYS = ("torch::kCUDA", "torch::kCPU", "torch::kMUSA", "torch::kPrivateUse1")


# ---------------------------------------------------------------------------
# 1. the mechanism
# ---------------------------------------------------------------------------


class TestDispatcherMechanism(unittest.TestCase):
    """The defect and the fix on throwaway ops. No wheel, no device."""

    @classmethod
    def setUpClass(cls):
        # One library for the whole class: re-defining a schema in the same
        # process is an error, and pytest may import this module once per run.
        cls.lib = torch.library.Library("sgl_test_518", "DEF")
        cls.lib.define("device_only_probe(int t) -> int")
        cls.lib.impl("device_only_probe", lambda t: t * 2, "CUDA")
        cls.lib.define("catch_all_probe(int t) -> int")
        cls.lib.impl("catch_all_probe", lambda t: t * 2, "CompositeExplicitAutograd")
        cls.lib.define("tensor_carrying_probe(Tensor x) -> int")
        cls.lib.impl("tensor_carrying_probe", lambda x: int(x.numel()), "CPU")

    def test_a_device_only_no_tensor_op_cannot_be_called(self):
        """The exact failure Gate A hit, on both arches."""
        with self.assertRaises(NotImplementedError) as caught:
            torch.ops.sgl_test_518.device_only_probe(3)
        message = str(caught.exception)
        self.assertIn("no tensor arguments", message)
        self.assertIn("no fallback function is registered", message)

    def test_the_catch_all_registration_is_callable(self):
        self.assertEqual(torch.ops.sgl_test_518.catch_all_probe(3), 6)

    def test_a_device_key_is_fine_once_a_tensor_carries_it(self):
        """The rule is about the SCHEMA, not about device-only registration in
        general: with a tensor to infer from, a device key routes normally."""
        self.assertEqual(
            torch.ops.sgl_test_518.tensor_carrying_probe(torch.zeros(7)), 7
        )


# ---------------------------------------------------------------------------
# 2. the shipped registrations
# ---------------------------------------------------------------------------


def _defs_and_impls(source: str):
    """``({name: schema}, {name: [key, ...]})`` for one TORCH_LIBRARY source.

    ``m.def`` arguments are frequently split across lines as adjacent string
    literals, so the literals are concatenated before parsing.
    """
    defs: dict[str, str] = {}
    for literals in re.findall(r"m\.def\(\s*((?:\"(?:[^\"\\]|\\.)*\"\s*)+)\)", source):
        schema = "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", literals))
        name = schema.split("(", 1)[0].strip()
        if name:
            defs[name] = schema
    impls: dict[str, list[str]] = {}
    for name, rest in re.findall(
        r"m\.impl\(\s*\"([^\"]+)\"\s*,\s*([^;]*?)\)\s*;", source, re.S
    ):
        key = rest.split(",")[0].strip() if "," in rest else ""
        impls.setdefault(name, []).append(key)
    return defs, impls


def _schema_has_tensor(schema: str) -> bool:
    args = schema[schema.find("(") + 1 : schema.rfind(") ->")]
    return "Tensor" in args


class TestRegistrationRatchet(unittest.TestCase):
    def test_no_tensorless_op_is_bound_to_a_device_key(self):
        offenders = []
        checked = 0
        for path in _EXTENSIONS:
            if not path.exists():
                continue
            defs, impls = _defs_and_impls(path.read_text())
            for name, schema in defs.items():
                if "->" not in schema or _schema_has_tensor(schema):
                    continue
                checked += 1
                for key in impls.get(name, []):
                    if key in _DEVICE_KEYS:
                        offenders.append(f"{path.name}: {name} bound to {key}")
        self.assertGreater(checked, 0, "parser found no tensorless schemas")
        self.assertEqual(
            offenders,
            [],
            "an op with no Tensor argument bound to a device dispatch key is "
            "unreachable -- the dispatcher has nothing to infer the backend "
            f"from and raises on every call: {offenders}",
        )

    def test_the_three_gguf_probes_are_the_ones_this_fixed(self):
        """Named, so the fix is legible and a rename cannot silently drop it."""
        src = _EXTENSIONS[0].read_text()
        _defs, impls = _defs_and_impls(src)
        for name in (
            "ggml_moe_get_block_size",
            "ggml_mmvq_kq_tuned",
            "ggml_mxfp4_native",
        ):
            with self.subTest(op=name):
                self.assertIn(name, impls, f"{name} lost its m.impl")
                self.assertEqual(impls[name], [""], f"{name} regained a device key")

    def test_the_parser_would_see_a_reintroduced_offender(self):
        """Can-discriminate: the ratchet is only worth its green if it goes red
        on the pre-#518 text."""
        pre_518 = (
            'm.def("ggml_moe_get_block_size(int type) -> int");\n'
            'm.impl("ggml_moe_get_block_size", torch::kCUDA, '
            "&ggml_moe_get_block_size);\n"
        )
        defs, impls = _defs_and_impls(pre_518)
        self.assertFalse(_schema_has_tensor(defs["ggml_moe_get_block_size"]))
        self.assertEqual(impls["ggml_moe_get_block_size"], ["torch::kCUDA"])

    def test_a_tensor_carrying_op_is_not_flagged(self):
        """The other half of can-discriminate: no false positives on the ~200
        device-keyed ops that do take tensors."""
        ok = (
            'm.def("ggml_moe_a8_vec(Tensor X, Tensor W, int type) -> Tensor");\n'
            'm.impl("ggml_moe_a8_vec", torch::kCUDA, &ggml_moe_a8_vec);\n'
        )
        defs, _impls = _defs_and_impls(ok)
        self.assertTrue(_schema_has_tensor(defs["ggml_moe_a8_vec"]))


# ---------------------------------------------------------------------------
# 3. the python-side fallback that carried the serving path meanwhile
# ---------------------------------------------------------------------------


class TestPythonMirrorStillWorks(unittest.TestCase):
    """The mirror must survive the fix: the wheel ships prebuilt and pinned by
    sha256, so a tree carrying #518 can still run a pre-#518 wheel."""

    def test_the_mirror_answers_when_the_op_raises(self):
        from sglang.srt.layers.quantization import gguf as G

        def _raise(_qtype):
            raise NotImplementedError("no dispatchable fallback (simulated)")

        # The name only exists on a host whose branch imported the kernels
        # (CUDA / MUSA); on a CPU box it is absent, so inject and restore
        # rather than assume.
        sentinel = object()
        original = getattr(G, "ggml_moe_get_block_size", sentinel)
        G.ggml_moe_get_block_size = _raise
        try:
            self.assertIn(G._ggml_moe_get_block_size(12), (4, 8))  # Q4_K
            self.assertEqual(G._ggml_moe_get_block_size(999), 0)  # unknown type
        finally:
            if original is sentinel:
                del G.ggml_moe_get_block_size
            else:
                G.ggml_moe_get_block_size = original


if __name__ == "__main__":
    unittest.main()
