# SPDX-License-Identifier: Apache-2.0
"""CPU-only guard rails for the HTCCL port (task #117).

These tests deliberately touch no GPU: they run on a machine whose cards are
busy, and they must also pass on the ROCm rank of a cross-vendor bring-up,
which has no CUDA at all.

What they lock down:
  1. the three HTCCL modules import without creating a CUDA context
     (a ROCm or CPU-only rank must be able to import them),
  2. the feature flag is OFF by default -> the dispatch path is byte-identical
     to stock sglang,
  3. the vendor-neutral hinge (_pin_host_memory) selects the runtime of the
     process it is running in, never the other vendor's,
  4. the calibration-persistence stub is rank-uniform (constant),
  5. every dispatch seam in parallel_state is guarded by `htccl_comm is not
     None`, i.e. flag-off cannot reach HTCCL code.
"""

import ast
import importlib.util
import pathlib
import sys
import types

import pytest
import torch

_COMM_DIR = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "distributed"
    / "device_communicators"
)
_PARALLEL_STATE = _COMM_DIR.parent / "parallel_state.py"
_ENVIRON = _COMM_DIR.parents[1] / "environ.py"


def _env_defaults():
    """Read the SGLANG_HTCCL* declarations out of THIS worktree's environ.py.

    Deliberately source-level rather than `from sglang.srt.environ import
    envs`: the venv is editable-installed against a different checkout, and
    importing sglang initializes CUDA. Both are disqualifying here.
    """
    tree = ast.parse(_ENVIRON.read_text())
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "Envs"
    )
    out = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign | ast.Assign):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("SGLANG_HTCCL"):
            continue
        call = node.value
        assert isinstance(call, ast.Call), f"{target.id} is not an Env* declaration"
        out[target.id] = (call.func.id, ast.literal_eval(call.args[0]))
    return out


def _load_standalone(name):
    """Import an HTCCL module directly from its file, bypassing the sglang
    package __init__ (which itself initializes CUDA)."""
    spec = importlib.util.spec_from_file_location(
        f"_htccl_test_{name}", _COMM_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1. import is CUDA-free
# --------------------------------------------------------------------------


def test_import_does_not_initialize_cuda():
    assert not torch.cuda.is_initialized(), (
        "precondition failed: something initialized CUDA before this test"
    )
    for name in ("htccl", "htccl_shm", "htccl_device"):
        _load_standalone(name)
        assert not torch.cuda.is_initialized(), (
            f"{name}.py created a CUDA context at import time; the ROCm rank "
            "of a cross-vendor group could not import it"
        )


def test_modules_expose_expected_api():
    assert hasattr(_load_standalone("htccl"), "HTCCLCommunicator")
    assert hasattr(_load_standalone("htccl_shm"), "HTCCLShmTransport")
    assert hasattr(_load_standalone("htccl_device"), "HTCCLDeviceTransport")


def test_communicator_implements_the_four_collectives():
    comm = _load_standalone("htccl").HTCCLCommunicator
    for op in ("all_reduce", "all_gather", "reduce_scatter", "broadcast", "close"):
        assert callable(getattr(comm, op, None)), f"missing {op}"


# --------------------------------------------------------------------------
# 2. flag defaults OFF
# --------------------------------------------------------------------------


def test_flag_is_off_by_default():
    kind, default = _env_defaults()["SGLANG_HTCCL"]
    assert (kind, default) == ("EnvBool", False), (
        "SGLANG_HTCCL must default to False -- backward compatibility "
        "requires the stock NCCL path when the flag is unset"
    )


def test_default_transport_is_device():
    assert _env_defaults()["SGLANG_HTCCL_TRANSPORT"] == ("EnvStr", "device")
    # ... and the module agrees, so env and code cannot drift apart.
    assert _load_standalone("htccl")._TRANSPORT == "device"


def test_all_htccl_envs_are_registered():
    """Every SGLANG_HTCCL* the ported modules read must be declared, so it
    shows up in the env dump and cannot be silently misspelled."""
    declared = set(_env_defaults())
    read = set()
    for name in ("htccl", "htccl_shm", "htccl_device"):
        for node in ast.walk(ast.parse((_COMM_DIR / f"{name}.py").read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("SGLANG_HTCCL"):
                    # f-string fragments carry trailing punctuation, e.g.
                    # f"SGLANG_HTCCL_RSAG_SHARES={override!r} ..." -> keep
                    # only the leading identifier.
                    ident = node.value.split("=")[0].split()[0].strip()
                    read.add(ident)
    assert read <= declared, f"env read but not declared: {sorted(read - declared)}"


# --------------------------------------------------------------------------
# 3. vendor-neutral hinge picks THIS process's runtime
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hip_version, expected_lib",
    [(None, "libcudart.so"), ("6.3.0", "libamdhip64.so")],
)
def test_pin_host_memory_selects_own_runtime(monkeypatch, hip_version, expected_lib):
    """The single vendor-specific call in HTCCL. Each process must register
    the shared segment with its OWN runtime -- that is what lets a CUDA and a
    ROCm process share one host mapping."""
    shm = _load_standalone("htccl_shm")
    monkeypatch.setattr(torch.version, "hip", hip_version, raising=False)

    requested = []

    class _FakeLib:
        def __getattr__(self, item):
            def _call(*args):
                return 0  # success

            _call.restype = None
            return _call

    def _fake_cdll(name):
        requested.append(name)
        return _FakeLib()

    monkeypatch.setattr(shm.ctypes, "CDLL", _fake_cdll)
    ok = shm._pin_host_memory(0x1000, 4096, torch.device("cpu"))

    assert ok is True
    assert requested == [expected_lib], (
        f"hip={hip_version!r} must load {expected_lib}, got {requested}"
    )


def test_pin_host_memory_reports_failure_without_raising(monkeypatch):
    """A failed host-register must degrade to unpinned, not crash: the device
    transport turns it into a clear error, the shm transport keeps going."""
    shm = _load_standalone("htccl_shm")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    def _boom(name):
        raise OSError("no such library")

    monkeypatch.setattr(shm.ctypes, "CDLL", _boom)
    assert shm._pin_host_memory(0x1000, 4096, torch.device("cpu")) is False


# --------------------------------------------------------------------------
# 4. calibration stub is rank-uniform
# --------------------------------------------------------------------------


def test_tune_stub_is_constant_and_rank_uniform():
    """Rank-divergent calibration deadlocks this fork's collectives rather
    than producing a wrong answer. A constant is trivially uniform."""
    dev = _load_standalone("htccl_device")
    for key in ("rsag_shares", "pipe_mib", "anything"):
        assert dev._tune_get(key) is None
    assert dev._tune_report("rsag_shares", "2,1,2") is None


# --------------------------------------------------------------------------
# 5. every seam is guarded -- flag-off cannot reach HTCCL
# --------------------------------------------------------------------------


def _htccl_attribute_uses(tree):
    """All `self.htccl_comm` loads outside __init__/destroy, with their line."""
    uses = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "htccl_comm"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            uses.append(node.lineno)
    return uses


def test_dispatch_seams_are_all_none_guarded():
    src = _PARALLEL_STATE.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()

    def _is_none_guard(test):
        """Match `self.htccl_comm is not None` and the getattr variant."""
        if not isinstance(test, ast.Compare) or not isinstance(
            test.ops[0], ast.IsNot
        ):
            return False
        return "htccl_comm" in ast.unparse(test.left)

    # Line numbers inside a body that a None-guard protects, plus the guard's
    # own condition line.
    guarded_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_none_guard(node.test):
            guarded_lines.add(node.test.lineno)
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if hasattr(sub, "lineno"):
                        guarded_lines.add(sub.lineno)

    # Lines where htccl_comm is assigned (construction / teardown), including
    # the annotated declaration `self.htccl_comm: Optional[Any] = None`.
    assigned_lines = {
        i + 1
        for i, line in enumerate(lines)
        if "self.htccl_comm =" in line or "self.htccl_comm:" in line
    }

    for lineno in _htccl_attribute_uses(tree):
        assert lineno in guarded_lines or lineno in assigned_lines, (
            f"{_PARALLEL_STATE}:{lineno} touches self.htccl_comm without an "
            "`is not None` guard -- flag-off would no longer be byte-identical"
        )


def test_seams_cover_the_four_collectives():
    """Guard against a seam silently disappearing in a rebase."""
    src = _PARALLEL_STATE.read_text()
    for op in (
        "self.htccl_comm.all_reduce(",
        "self.htccl_comm.reduce_scatter(",
        "self.htccl_comm.all_gather(",
        "self.htccl_comm.broadcast(",
    ):
        assert op in src, f"dispatch seam missing: {op}"


def test_construction_is_flag_gated():
    src = _PARALLEL_STATE.read_text()
    assert "if envs.SGLANG_HTCCL.get() and self.world_size > 1:" in src, (
        "HTCCL must only be constructed when the flag is on"
    )
    # The communicator module must not be imported at all when the flag is off.
    assert src.count("from sglang.srt.distributed.device_communicators.htccl import") == 1
    idx_gate = src.index("if envs.SGLANG_HTCCL.get()")
    idx_import = src.index(
        "from sglang.srt.distributed.device_communicators.htccl import"
    )
    assert idx_import > idx_gate, "the HTCCL import must sit inside the flag gate"
