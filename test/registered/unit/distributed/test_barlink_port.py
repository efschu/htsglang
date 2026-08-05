# SPDX-License-Identifier: Apache-2.0
"""CPU-only guard rails for the barlink port (task #117).

These tests deliberately touch no GPU: they run on a machine whose cards are
busy, and they must also pass on the ROCm rank of a cross-vendor bring-up,
which has no CUDA at all.

What they lock down:
  1. the three barlink modules import without creating a CUDA context
     (a ROCm or CPU-only rank must be able to import them),
  2. the feature flag is OFF by default -> the dispatch path is byte-identical
     to stock sglang,
  3. the vendor-neutral hinge (_pin_host_memory) selects the runtime of the
     process it is running in, never the other vendor's,
  4. the calibration-persistence stub is rank-uniform (constant),
  5. every dispatch seam in parallel_state is guarded by `barlink_comm is not
     None`, i.e. flag-off cannot reach barlink code.
"""

import ast
import importlib.util
import inspect
import pathlib
import sys
import types

import pytest
import torch
import torch.distributed

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
    """Read the SGLANG_BARLINK* declarations out of THIS worktree's environ.py.

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
        if not isinstance(target, ast.Name) or not target.id.startswith("SGLANG_BARLINK"):
            continue
        call = node.value
        assert isinstance(call, ast.Call), f"{target.id} is not an Env* declaration"
        out[target.id] = (call.func.id, ast.literal_eval(call.args[0]))
    return out


def _load_standalone(name):
    """Import an barlink module directly from its file, bypassing the sglang
    package __init__ (which itself initializes CUDA)."""
    spec = importlib.util.spec_from_file_location(
        f"_barlink_test_{name}", _COMM_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1. import is CUDA-free
# --------------------------------------------------------------------------


def test_import_does_not_initialize_cuda():
    """Importing the barlink modules must create no CUDA context.

    Runs in a FRESH interpreter on purpose. Asserting
    ``not torch.cuda.is_initialized()`` in-process makes the result depend on
    whatever else the pytest session imported first -- any earlier test in this
    directory that touches CUDA turns this into a spurious red, and a test that
    is only green when run alone is a test nobody can trust. A subprocess makes
    the precondition true by construction, and it checks the identical property:
    a ROCm or CPU-only rank of a cross-vendor group must be able to import
    these modules.
    """
    import subprocess
    import textwrap

    probe = textwrap.dedent(
        """
        import importlib.util, sys, pathlib
        import torch

        comm_dir = pathlib.Path(sys.argv[1])
        assert not torch.cuda.is_initialized(), "torch import alone initialized CUDA"
        for name in ("barlink", "barlink_shm", "barlink_device", "barlink_host"):
            spec = importlib.util.spec_from_file_location(
                f"_barlink_probe_{name}", comm_dir / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            if torch.cuda.is_initialized():
                raise SystemExit(
                    f"{name}.py created a CUDA context at import time; the "
                    "ROCm rank of a cross-vendor group could not import it"
                )
        print("OK")
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", probe, str(_COMM_DIR)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0 and "OK" in res.stdout, (
        f"barlink import probe failed (rc={res.returncode}):\n"
        f"stdout: {res.stdout}\nstderr: {res.stderr}"
    )


def test_modules_expose_expected_api():
    assert hasattr(_load_standalone("barlink"), "BarlinkCommunicator")
    assert hasattr(_load_standalone("barlink_shm"), "BarlinkShmTransport")
    assert hasattr(_load_standalone("barlink_device"), "BarlinkDeviceTransport")


def test_communicator_implements_the_supported_collectives():
    comm = _load_standalone("barlink").BarlinkCommunicator
    for op in (
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "broadcast",
        "close",
    ):
        assert callable(getattr(comm, op, None)), f"missing {op}"


def test_out_parameter_forms_add_no_new_collective():
    """The out-parameter forms must be compositions of the existing ops.

    A new collective would need its own rank-uniformity argument; composing
    keeps the existing one intact. Assert they only call sibling methods and
    never torch.distributed directly.
    """
    src = (_COMM_DIR / "barlink.py").read_text()
    tree = ast.parse(src)
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "BarlinkCommunicator"
    )
    for name in ("all_gather_into_tensor", "reduce_scatter_tensor"):
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        body = ast.unparse(fn)
        assert "dist." not in body, (
            f"{name} calls torch.distributed directly; it must compose the "
            "existing barlink collectives instead"
        )


def _fake_comm(module, slot_bytes=1 << 30):
    """An BarlinkCommunicator wired for a real transport seam, on CPU tensors.

    __init__ is bypassed on purpose: it builds CUDA streams and a real process
    group, neither of which this file is allowed to touch. The fake transport
    reproduces the shm transport's calling convention EXACTLY -- borrow
    `comm._get_out_buf`, copy in, reduce in place -- because that is the path
    the out-of-place contract has to hold across. `_out_pool` is supplied so a
    reintroduced shape-keyed cache runs to completion and fails on the
    assertions below rather than erroring on a missing attribute.
    """

    class _Transport:
        def __init__(self):
            self.slot_bytes = slot_bytes

        @staticmethod
        def handles(op, nbytes):
            return op == "all_reduce"

        @staticmethod
        def barlink_all_reduce(comm, inp):
            out = comm._get_out_buf(inp)
            out.copy_(inp)
            out.mul_(2)  # stand-in for the peers' contribution
            return out

    comm = object.__new__(module.BarlinkCommunicator)
    comm.disabled = False
    comm.transport = _Transport()
    comm._out_pool = {}
    return comm


def test_all_reduce_is_out_of_place_across_calls():
    """Two same-shape results must be two DISTINCT tensors.

    `all_reduce` documents itself as out-of-place ("returns a new tensor").
    A per-(shape, dtype) buffer cache silently broke that: the second call
    handed back the very tensor the first result lived in, so a caller
    holding both saw its first result overwritten. In the server this
    destroyed the model forward -- valid HTTP 200s, garbage tokens, no
    crash and no hang -- on every transport that used this helper.
    """
    module = _load_standalone("barlink")
    comm = _fake_comm(module)

    x = torch.ones(4, 8)
    y = torch.full((4, 8), 3.0)

    first = comm.all_reduce(x)
    first_snapshot = first.clone()
    second = comm.all_reduce(y)

    assert first.data_ptr() != second.data_ptr(), (
        "all_reduce returned the same storage twice -- the out-of-place "
        "contract is broken and the earlier result has been clobbered"
    )
    assert torch.equal(first, first_snapshot), (
        "the first all_reduce result changed while a later all_reduce ran; "
        "callers that hold two same-shape results read corrupted data"
    )
    assert torch.equal(second, y * 2)


def test_communicator_keeps_no_shape_keyed_output_cache():
    """Guard the fix at source level, not just behaviourally.

    The graph-capturable transport (`barlink_device`) allocates its output
    fresh per call; the CPU transports cannot be captured at all. So no
    barlink path has a reason to hold a persistent output buffer, and
    reintroducing one would resurrect the corruption above.
    """
    src = (_COMM_DIR / "barlink.py").read_text()
    tree = ast.parse(src)
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "BarlinkCommunicator"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_out_buf"
    )
    body = ast.unparse(fn)
    assert "_out_pool" not in ast.unparse(cls), (
        "BarlinkCommunicator reintroduced a persistent output buffer cache"
    )
    assert "empty_like" in body, (
        "_get_out_buf must allocate a fresh tensor per call"
    )


def test_device_extension_builds_for_every_arch_and_keys_the_cache_by_it(
    monkeypatch,
    tmp_path,
):
    """A mixed-arch rig must not share a single-arch cubin between ranks.

    Original defect: `load_inline` with no arch flags and a name-only cache
    key; under per-rank CUDA_VISIBLE_DEVICES isolation each worker sees ONE
    card, so the first compiler fixed the arch for everyone and the sm_86
    ranks died on their first kernel launch. Both halves matter: the flags
    make the binary fat, the arch-aware NAME stops a stale single-arch build
    directory from being reused. The seam is now vendor-aware
    (gfx900 merge): same invariants, keyed per (vendor, arches).
    """
    import torch.utils.cpp_extension as cpp_extension

    # _load_ext now does cache hygiene (#181) on the real torch extensions
    # root before building, so give this test a root of its own: a unit test
    # must not sweep, mark, or otherwise write into the developer's warm
    # extension cache.
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))

    module = _load_standalone("barlink_device")
    module._ext = None
    monkeypatch.setattr(
        module, "_resolve_build_arches",
        lambda cpu_group: {"cuda": ["8.6", "12.0"]},
    )
    monkeypatch.setattr(module, "_local_vendor", lambda: "cuda")

    captured = {}

    def fake_load_inline(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(barlink_allreduce=None, barlink_allgather=None)

    monkeypatch.setattr(cpp_extension, "load_inline", fake_load_inline)
    module._load_ext(cpu_group=None)

    name = captured["name"]
    assert "cuda" in name, f"{name!r} does not encode the vendor"
    assert "86" in name and "120" in name, (
        f"extension name {name!r} does not encode the architecture set -- a "
        "cached single-arch .so from an earlier run would be reused"
    )
    flags = " ".join(captured.get("extra_cuda_cflags") or [])
    for sm in ("86", "120"):
        assert f"code=sm_{sm}" in flags, (
            f"no SASS requested for sm_{sm}; flags were {flags!r}"
        )


def test_device_extension_arch_union_keeps_vendors_separate(monkeypatch):
    """The group union is per (vendor, arch) pairs, never bare capabilities.

    Two invariants in one: (a) a rank that sees only its own card still
    builds for the WHOLE group's arches of its vendor (the mixed-NVIDIA
    case); (b) '9.0' from a Hopper rank and gfx900 from a Vega rank must
    never merge -- the numeric namespaces collide, which is the third
    instance of the (9,0) collision the vendor-aware key exists for.
    """
    module = _load_standalone("barlink_device")

    monkeypatch.setattr(module, "_local_vendor", lambda: "cuda")
    monkeypatch.setattr(module, "_local_arches", lambda vendor: ["8.6"])

    def fake_all_gather_object(out_list, obj, group=None):
        out_list[0] = obj                       # this rank: cuda 8.6
        out_list[1] = [("cuda", "12.0")]        # the 5090 rank
        out_list[2] = [("hip", "gfx900")]       # a Vega rank

    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group: 3, raising=False
    )
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", fake_all_gather_object,
        raising=False,
    )

    by_vendor = module._resolve_build_arches(cpu_group=None)
    assert by_vendor.get("cuda") == ["8.6", "12.0"], (
        f"got {by_vendor}: a rank must build for the whole group's arches "
        "of its vendor"
    )
    assert by_vendor.get("hip") == ["gfx900"], by_vendor
    # the collision case: no flat set may ever mix the two namespaces
    assert "gfx900" not in by_vendor.get("cuda", []), by_vendor

    # per-vendor flags stay in their own compiler's namespace
    nv = " ".join(module._build_flags("cuda", ["8.6", "12.0"]))
    amd = " ".join(module._build_flags("hip", ["gfx900"]))
    assert "-gencode" in nv and "--offload-arch" not in nv
    assert "--offload-arch=gfx900" in amd and "-gencode" not in amd


# --------------------------------------------------------------------------
# 2. flag defaults OFF
# --------------------------------------------------------------------------


def test_flag_is_off_by_default():
    kind, default = _env_defaults()["SGLANG_BARLINK"]
    assert (kind, default) == ("EnvBool", False), (
        "SGLANG_BARLINK must default to False -- backward compatibility "
        "requires the stock NCCL path when the flag is unset"
    )


def test_default_transport_is_device():
    assert _env_defaults()["SGLANG_BARLINK_TRANSPORT"] == ("EnvStr", "device")
    # ... and the module agrees, so env and code cannot drift apart.
    assert _load_standalone("barlink")._TRANSPORT == "device"


def test_all_barlink_envs_are_registered():
    """Every SGLANG_BARLINK* the ported modules read must be declared, so it
    shows up in the env dump and cannot be silently misspelled."""
    declared = set(_env_defaults())
    read = set()
    for name in ("barlink", "barlink_shm", "barlink_device", "barlink_host"):
        for node in ast.walk(ast.parse((_COMM_DIR / f"{name}.py").read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("SGLANG_BARLINK"):
                    # f-string fragments carry trailing punctuation, e.g.
                    # f"SGLANG_BARLINK_RSAG_SHARES={override!r} ..." -> keep
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
    """The single vendor-specific call in barlink. Each process must register
    the shared segment with its OWN runtime -- that is what lets a CUDA and a
    ROCm process share one host mapping."""
    shm = _load_standalone("barlink_shm")
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
    shm = _load_standalone("barlink_shm")
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
    dev = _load_standalone("barlink_device")
    for key in ("rsag_shares", "pipe_mib", "anything"):
        assert dev._tune_get(key) is None
    assert dev._tune_report("rsag_shares", "2,1,2") is None


# --------------------------------------------------------------------------
# 5. every seam is guarded -- flag-off cannot reach barlink
# --------------------------------------------------------------------------


def _barlink_attribute_uses(tree):
    """All `self.barlink_comm` loads outside __init__/destroy, with their line."""
    uses = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "barlink_comm"
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
        """Match `self.barlink_comm is not None`, including as one operand of a
        boolean condition (`barlink_comm is not None or pynccl is None ...`)."""
        if isinstance(test, ast.BoolOp):
            return any(_is_none_guard(v) for v in test.values)
        if not isinstance(test, ast.Compare) or not isinstance(
            test.ops[0], ast.IsNot
        ):
            return False
        return "barlink_comm" in ast.unparse(test.left)

    # Line numbers inside a body that a None-guard protects, plus the guard's
    # own condition line.
    guarded_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_none_guard(node.test):
            for sub in ast.walk(node.test):
                if hasattr(sub, "lineno"):
                    guarded_lines.add(sub.lineno)
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if hasattr(sub, "lineno"):
                        guarded_lines.add(sub.lineno)

    # Lines where barlink_comm is assigned (construction / teardown), including
    # the annotated declaration `self.barlink_comm: Optional[Any] = None`.
    assigned_lines = {
        i + 1
        for i, line in enumerate(lines)
        if "self.barlink_comm =" in line or "self.barlink_comm:" in line
    }

    # A touch that IS the guard itself, outside an `if` -- e.g.
    #     _barlink_active = self.barlink_comm is not None
    # This is not a loophole: the invariant being protected is that flag-off
    # falls through, and such an expression evaluates to False exactly when the
    # flag is off, which is the required behaviour. What stays forbidden is an
    # unguarded USE of the object (attribute access, call, truthiness), because
    # that is what could change flag-off semantics.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.ops[0], ast.IsNot)
            and "barlink_comm" in ast.unparse(node.left)
        ):
            for sub in ast.walk(node):
                if hasattr(sub, "lineno"):
                    guarded_lines.add(sub.lineno)

    for lineno in _barlink_attribute_uses(tree):
        assert lineno in guarded_lines or lineno in assigned_lines, (
            f"{_PARALLEL_STATE}:{lineno} touches self.barlink_comm without an "
            "`is not None` guard -- flag-off would no longer be byte-identical"
        )


def test_seams_cover_every_supported_collective():
    """Guard against a seam silently disappearing in a rebase."""
    src = _PARALLEL_STATE.read_text()
    for op in (
        "self.barlink_comm.all_reduce(",
        "self.barlink_comm.reduce_scatter(",
        "self.barlink_comm.reduce_scatter_tensor(",
        "self.barlink_comm.all_gather(",
        "self.barlink_comm.all_gather_into_tensor(",
        "self.barlink_comm.broadcast(",
    ):
        assert op in src, f"dispatch seam missing: {op}"


# --------------------------------------------------------------------------
# 6. unsupported collectives fail fast rather than reaching NCCL
# --------------------------------------------------------------------------

# On a mixed-vendor group a collective that reaches NCCL does not run slowly,
# it deadlocks -- NCCL and RCCL cannot form a joint communicator. Every
# collective barlink does not implement must therefore raise immediately.
_MUST_FAIL_FAST = (
    "reduce_scatterv",
    "all_gatherv",
    "reduce_scatter(output, input_list)",
    "all_gather(output_tensor_list=...)",
)


def test_unsupported_collectives_are_guarded():
    src = _PARALLEL_STATE.read_text()
    for op in _MUST_FAIL_FAST:
        assert f'self._barlink_unsupported("{op}")' in src, (
            f"{op} has no barlink fail-fast guard -- under SGLANG_BARLINK it "
            "would silently fall through to NCCL and hang"
        )


def test_fail_fast_helper_raises_and_names_the_op():
    """The message must name the op; a bare exception minutes into a run is
    what this guard exists to avoid."""
    tree = ast.parse(_PARALLEL_STATE.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_barlink_unsupported"
    )
    body = ast.unparse(fn)
    assert "raise NotImplementedError" in body
    assert "{op!r}" in body, "the error message must interpolate the op name"


def test_async_allgather_does_not_bypass_barlink():
    """cp_all_gather_into_tensor_async has a pynccl fast path; under barlink it
    must route to the barlink-aware synchronous form."""
    tree = ast.parse(_PARALLEL_STATE.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "cp_all_gather_into_tensor_async"
    )
    body = ast.unparse(fn)
    assert "self.barlink_comm is not None" in body, (
        "cp_all_gather_into_tensor_async would take the pynccl path while "
        "barlink is active"
    )


def test_construction_is_flag_gated():
    """The gate is `should_build_barlink`, and its body is still the flag.

    It used to be inlined here. #598 gave it a name because a SECOND reader
    needs it: the VRAM ledger has to answer "does this launch build an NCCL
    communicator at all?" during argument parsing, and a ledger that re-derived
    `envs.SGLANG_BARLINK.get() and world_size > 1` would keep pricing that term
    at 0 after this condition changed. So the assertion moves down one level --
    the gate must BE the shared predicate, and the predicate must still be the
    flag -- rather than being dropped.
    """
    from sglang.srt.distributed.parallel_state import should_build_barlink

    src = _PARALLEL_STATE.read_text()
    assert "if should_build_barlink(self.world_size):" in src, (
        "barlink must only be constructed when the flag is on"
    )
    assert (
        "return bool(envs.SGLANG_BARLINK.get()) and world_size > 1"
        in inspect.getsource(should_build_barlink)
    ), "should_build_barlink must still be exactly the flag-and-multi-rank gate"
    # The communicator module must not be imported at all when the flag is off.
    assert src.count("from sglang.srt.distributed.device_communicators.barlink import") == 1
    idx_gate = src.index("if should_build_barlink(self.world_size):")
    idx_import = src.index(
        "from sglang.srt.distributed.device_communicators.barlink import"
    )
    assert idx_import > idx_gate, "the barlink import must sit inside the flag gate"


def test_cpu_transports_are_rejected_while_cuda_graphs_are_on():
    """Host-staged transports + CUDA graphs must fail at STARTUP, not later.

    Every non-device transport host-stages its collectives (shm: two
    cudaStreamSynchronize per op plus a spin on shm counters; gloo: a
    cudaEventSynchronize and a gloo CPU collective per chunk; ucx: blocking
    device-boundary copies through pinned host buffers). Inside a capture
    that raises `cudaErrorStreamCaptureUnsupported` from whichever kernel is
    capturing -- observed on arm E, where it read as an unrelated CUDA
    fault. The constraint used to live only in a log line.

    "ucx" is in this list because its absence was a measurement-integrity
    hole (task #246): a ucx cross-rig boot with graphs enabled passed the
    guard and then captured at most rank-local regions, so nobody could tell
    which regime a measurement ran in. The guard is an ALLOWLIST of the
    capturable transports, so an unknown name -- which silently falls back
    to the host-staged gloo plane in barlink._build_transport -- is rejected
    too, instead of repeating the ucx gap for the next transport.
    """
    import types as _types

    import sglang.srt.distributed.parallel_state as ps

    graphs_on = _types.SimpleNamespace(disable_cuda_graph=False)
    graphs_off = _types.SimpleNamespace(disable_cuda_graph=True)

    import sglang.srt.runtime_context as rc
    orig = rc.get_server_args
    try:
        rc.get_server_args = lambda: graphs_on
        for transport in ("shm", "gloo", "ucx", "some-future-transport"):
            with pytest.raises(ValueError) as ei:
                ps._enforce_cpu_transport_needs_eager(transport)
            msg = str(ei.value)
            assert "--disable-cuda-graph" in msg, msg
            assert "device" in msg, msg  # names the capturable alternative
        # the GPU-driven transport is capturable and must NOT be rejected
        ps._enforce_cpu_transport_needs_eager("device")
        # the allowlist and the registry must agree on what "capturable"
        # means -- a transport registered as capturable but unknown to the
        # registry (or vice versa) is how the ucx gap happened
        from sglang.srt.distributed.device_communicators.barlink import (
            TRANSPORT_REGISTRY,
        )

        assert ps.CAPTURABLE_BARLINK_TRANSPORTS <= set(TRANSPORT_REGISTRY)

        rc.get_server_args = lambda: graphs_off
        for transport in ("shm", "gloo", "ucx", "device"):
            ps._enforce_cpu_transport_needs_eager(transport)
    finally:
        rc.get_server_args = orig


def _reduce_scatter_reference(total, world, rank, dim):
    """What reduce_scatter must return: sum over ranks, then this rank's slice
    ALONG `dim` -- computed independently of the implementation."""
    chunk = total.shape[dim] // world
    return total.narrow(dim, rank * chunk, chunk)


def test_reduce_scatter_slices_the_requested_axis():
    """KNOWN-ANSWER test on VALUES, not shapes.

    `movedim(0, dim)` moves axis 0 TO position dim; it does not bring axis dim
    to the front. The two coincide only for dim in {0, 1}, so from dim >= 2 the
    old code left the ORIGINAL axis 1 in front and scattered THAT, while every
    shape assertion still passed. The signature defaults to dim=-1, so a bare
    reduce_scatter(x) on ndim >= 3 silently distributed the wrong axis.
    2-D is accidentally correct, which is why it survived.

    Elements are encoded so that any transposition or wrong-axis slice shows up
    as a value mismatch, not merely a shape mismatch.
    """
    module = _load_standalone("barlink")

    for world in (2, 3, 4):
        for shape, dim in (
            ((4, 6, 2), 2),
            ((2, 4, 6, 2), 3),
            ((6, 4, 2), 0),
            ((4, 6, 2), 1),
            ((12, 5), -1),
            ((12, 5), 0),
        ):
            ndim = len(shape)
            d = dim % ndim
            if shape[d] % world:
                continue
            # distinct value per position: no transposition can survive this
            total = torch.arange(1, 1 + torch.tensor(shape).prod().item(),
                                 dtype=torch.float32).reshape(shape)

            for rank in range(world):
                comm = object.__new__(module.BarlinkCommunicator)
                comm.disabled = False
                comm.world_size = world
                comm.rank = rank
                comm.transport = None  # force the inline path; axis logic under test
                # all_reduce is not under test here: hand back the summed
                # tensor directly so the axis logic is isolated.
                comm.all_reduce = lambda x, _t=total: _t.clone()

                got = module.BarlinkCommunicator.reduce_scatter(
                    comm, total.clone(), dim
                )
                want = _reduce_scatter_reference(total, world, rank, d)
                assert got.shape == want.shape, (
                    f"world={world} shape={shape} dim={dim} rank={rank}: "
                    f"shape {tuple(got.shape)} != {tuple(want.shape)}"
                )
                assert torch.equal(got, want), (
                    f"world={world} shape={shape} dim={dim} rank={rank}: "
                    f"reduce_scatter returned the wrong AXIS's data "
                    f"(shape matched, values did not)"
                )
