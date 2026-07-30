from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import logging
import os
import pathlib
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import torch

from sglang.jit_kernel.cache_health import (
    PathLike,
    building_marker,
    heal_entry,
    purge_entry,
    sweep_cache_root,
)
from sglang.utils import is_in_ci

if TYPE_CHECKING:
    from tvm_ffi import Module

F = TypeVar("F", bound=Callable[..., Any])
_FULL_TEST_ENV_VAR = "SGLANG_JIT_KERNEL_RUN_FULL_TESTS"

logger = logging.getLogger(__name__)


def should_run_full_tests() -> bool:
    return os.getenv(_FULL_TEST_ENV_VAR, "false").lower() == "true"


def get_ci_test_range(full_range: List[Any], ci_range: List[Any]) -> List[Any]:
    if should_run_full_tests():
        return full_range
    return ci_range if is_in_ci() else full_range


def cache_once(fn: F) -> F:
    """
    NOTE: `functools.lru_cache` is not compatible with `torch.compile`
    So we manually implement a simple cache_once decorator to replace it.
    """
    result_map = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore


def cache_once_per_arch(fn: F) -> F:
    """`cache_once` for a value that is only valid on ONE GPU architecture.

    A JIT module is a single-arch artefact: ``sgl_kernel/utils.cuh`` has
    ``static_assert(__CUDA_ARCH__ == SGL_CUDA_ARCH)``, so one build targets one
    compute capability and one only -- there is no multi-arch fatbin to fall
    back on. A memo that ignores the architecture therefore hands the FIRST
    device's cubin to every later device in the same process.

    That is not hypothetical. The card probe measures every GPU of a rig from
    one process; on the reference rig (RTX 5090 + 2x RTX 3080) it loaded the
    sm_120 ``gptq_marlin_repack`` module on device 0, kept it in a
    process-global memo, and then launched it on the two sm_86 cards:

        CUDA error: no kernel image is available for execution on the device

    Serving ranks are isolated to one device each, so there the per-arch key
    collapses to the per-process key this replaces and nothing changes. It is
    every process that legitimately touches more than one architecture -- the
    probe, the planner sweep, any non rank-isolated multi-GPU tool -- that
    needs the architecture in the key.

    Cost is one ``torch.cuda.current_device()`` per call. Dynamo constant-folds
    that call and installs an EQUALS_MATCH guard on it (it is in
    ``trace_rules``' in-graph set), so this stays ``torch.compile``-safe for the
    same reason ``cache_once`` is.
    """
    result_map: Dict[Any, Any] = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (
            get_jit_cuda_arch().target_name,
            args,
            tuple(sorted(kwargs.items())),
        )
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    wrapper.arch_scoped_cache = result_map  # type: ignore[attr-defined]
    return wrapper  # type: ignore


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


_QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_ANGLE_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*<(sgl_kernel/[^>]+)>", re.MULTILINE)


def _local_jit_source_hash(source_files: List[str]) -> str:
    """Hash JIT source contents so TVM-FFI cache keys track included headers."""
    digest = hashlib.sha256()
    seen: set[pathlib.Path] = set()
    stack = [pathlib.Path(path).resolve() for path in source_files]
    include_dir = KERNEL_PATH / "include"

    while stack:
        path = stack.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)

        data = path.read_bytes()
        # Relative to kernel root, not absolute: the key must track source
        # content, not install location (differs across runners / job dirs).
        try:
            ident = str(path.relative_to(KERNEL_PATH))
        except ValueError:
            ident = path.name
        digest.update(ident.encode())
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")

        text = data.decode("utf-8", errors="ignore")
        for include in _QUOTED_INCLUDE_RE.findall(text):
            include_path = (path.parent / include).resolve()
            if include_path.is_file():
                stack.append(include_path)
        for include in _ANGLE_INCLUDE_RE.findall(text):
            include_path = (include_dir / include).resolve()
            if include_path.is_file():
                stack.append(include_path)

    return digest.hexdigest()[:16]


def _build_input_hash(
    *,
    arch: str,
    vendor: str,
    backend: str,
    header_only: bool,
    cpp_wrappers: Sequence[Tuple[str, str]],
    cuda_wrappers: Sequence[Tuple[str, str]],
    extra_cflags: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    dependencies: Sequence[str],
) -> str:
    """Digest of every build input that is NOT source text.

    ``_local_jit_source_hash`` covers the sources; this covers the rest of what
    decides the emitted binary -- the target architecture, the toolchain, the
    exported wrappers and the full flag lists. Template parameters reach nvcc
    as ``-D`` flags at several call sites, so two builds can share every source
    byte and still be different kernels; before this, they shared a cache entry.

    Deliberately NOT in the digest: ``extra_include_paths``. Those are absolute
    and differ between a checkout and a wheel install, and folding them in would
    undo the property ``_local_jit_source_hash`` exists for -- a key that tracks
    content, not install location. They are recorded in the provenance file
    instead, where a mismatch is visible to a human without partitioning the
    cache.
    """
    digest = hashlib.sha256()
    for part in (
        arch,
        vendor,
        backend,
        "header_only" if header_only else "exported",
    ):
        digest.update(part.encode())
        digest.update(b"\0")
    for group in (
        [f"{a}={b}" for a, b in cpp_wrappers],
        [f"{a}={b}" for a, b in cuda_wrappers],
        list(extra_cflags),
        list(extra_cuda_cflags),
        list(extra_ldflags),
        sorted(dependencies),
    ):
        for item in group:
            digest.update(str(item).encode())
            digest.update(b"\0")
        digest.update(b"\1")
    return digest.hexdigest()[:12]


#: Written next to the artefact by every build this module performs. The cache
#: root is shared by every checkout, worktree and wheel install on a host, so
#: reuse must be VERIFIED rather than assumed from a directory name.
PROVENANCE_NAME = "sgl_jit_provenance.json"

#: Bumped when the record's meaning changes; an older record is not trusted.
PROVENANCE_VERSION = 1


def provenance_check_enabled() -> bool:
    """Operator escape hatch, read per call like the self-heal switch.

    Turning it off restores the old behaviour -- reuse whatever sits under the
    key -- which is the wrong default but the right lever to have if the check
    ever misjudges an entry on a host that cannot afford a rebuild.
    """
    return os.environ.get("SGLANG_JIT_PROVENANCE_CHECK", "1") not in (
        "0",
        "false",
        "False",
    )


def _provenance_record(
    *,
    module_name: str,
    source_hash: str,
    build_hash: str,
    arch: str,
    vendor: str,
    include_paths: Sequence[str],
) -> Dict[str, Any]:
    return {
        "version": PROVENANCE_VERSION,
        "module": module_name,
        "source_hash": source_hash,
        "build_hash": build_hash,
        # One entry today, and the list is the point: the artefact declares the
        # complete set of architectures it can run on, so a reader never has to
        # infer it from the directory name. It holds exactly one element because
        # SGL_CUDA_ARCH is a single macro that sgl_kernel/utils.cuh
        # static_asserts against __CUDA_ARCH__ -- a multi-arch fatbin cannot be
        # built here. Should that constraint ever lift, the check below already
        # accepts a superset.
        "target_archs": [arch],
        "vendor": vendor,
        "tvm_ffi": _tvm_ffi_version(),
        "source_tree": str(KERNEL_PATH),
        "include_paths": list(include_paths),
        "host": _provenance_host(),
        "pid": os.getpid(),
        "built_at": time.time(),
    }


def _provenance_host() -> str:
    try:
        return os.uname().nodename
    except AttributeError:  # pragma: no cover - non-POSIX
        import socket

        return socket.gethostname()


def _write_provenance(build_directory: str, record: Dict[str, Any]) -> None:
    path = pathlib.Path(build_directory) / PROVENANCE_NAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a reader must never see a half-written record and
        # conclude the artefact is unverifiable.
        tmp = path.with_name(f"{PROVENANCE_NAME}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True))
        tmp.replace(path)
    except OSError as exc:  # never fail a build on bookkeeping
        logger.warning("Could not record JIT provenance for %s: %r", path.parent, exc)


def check_provenance(
    build_directory: PathLike,
    record: Dict[str, Any],
) -> Tuple[bool, str]:
    """May the artefact in ``build_directory`` be reused for ``record``?

    Returns ``(ok, reason)``. ``reason`` is always populated: on acceptance it
    names the tree the artefact was built from, which is what makes a foreign
    ``__FILE__`` in a later runtime error explainable instead of alarming.
    """
    path = pathlib.Path(build_directory) / PROVENANCE_NAME
    try:
        stored = json.loads(path.read_text())
    except FileNotFoundError:
        return False, (
            "no provenance record -- the entry predates provenance tracking or "
            "was written by a build that did not finish"
        )
    except (OSError, ValueError) as exc:
        return False, f"provenance record unreadable ({exc!r})"

    if not isinstance(stored, dict) or stored.get("version") != PROVENANCE_VERSION:
        return False, f"provenance version {stored!r} is not {PROVENANCE_VERSION}"

    for field in ("module", "source_hash", "build_hash", "vendor"):
        if stored.get(field) != record[field]:
            return False, (
                f"{field} mismatch: artefact {stored.get(field)!r}, "
                f"this process {record[field]!r}"
            )

    wanted = set(record["target_archs"])
    have = set(stored.get("target_archs") or [])
    if not wanted <= have:
        return False, (
            f"architecture mismatch: artefact targets {sorted(have)}, this "
            f"process needs {sorted(wanted)}"
        )

    return True, f"built from {stored.get('source_tree')!r}"


@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    cur_dir = pathlib.Path(__file__).parent.resolve()

    # first, try this directory structure
    def _environment_install():
        candidate = cur_dir.resolve()
        if (candidate / "include").exists() and (candidate / "csrc").exists():
            return candidate
        return None

    def _package_install():
        # TODO: support find path by package
        return None

    path = _environment_install() or _package_install()
    if path is None:
        raise RuntimeError("Cannot find sglang.jit_kernel path")
    return path


KERNEL_PATH = _resolve_kernel_path()
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_LDFLAGS = []
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, str, bool, torch.dtype]


class CPPArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float: "fp32_t",
    torch.float16: "fp16_t",
    torch.float8_e4m3fn: "fp8_e4m3_t",
    torch.bfloat16: "bf16_t",
    torch.int8: "int8_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}


# AMD/ROCm note:
@cache_once
def is_hip_runtime() -> bool:
    return bool(torch.version.hip)


# MThreads/MUSA note:
@cache_once
def is_musa_runtime() -> bool:
    return hasattr(torch.version, "musa") and torch.version.musa is not None


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, str, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)


@cache_once
def _tvm_ffi_version() -> str:
    try:
        import tvm_ffi

        version = getattr(tvm_ffi, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        from importlib.metadata import version as dist_version

        return dist_version("apache-tvm-ffi")
    except Exception:
        return "unknown"


def _jit_cache_root() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
    ).expanduser()


def _selfheal_enabled() -> bool:
    # Read per call, not at import: the flag exists so an operator can turn the
    # sweep off in place if it ever misjudges an entry.
    return os.environ.get("SGLANG_JIT_CACHE_SELFHEAL", "1") not in ("0", "false", "False")


_jit_cache_swept = False


def _selfheal_jit_cache_once() -> List[str]:
    """Sweep incomplete entries out of the JIT cache, once per process.

    Runs on the first load_jit call rather than from server startup: that is
    the earliest point at which the cache is about to be trusted, needs no
    wiring into every entrypoint, and is still before any kernel is built.
    Idempotent, and a peer rank's in-flight build directory is skipped (see
    cache_health.entry_state).
    """
    global _jit_cache_swept
    if _jit_cache_swept:
        return []
    _jit_cache_swept = True
    if not _selfheal_enabled():
        return []
    try:
        return sweep_cache_root(_jit_cache_root())
    except Exception as exc:  # never fail a boot on cache hygiene
        logger.warning("JIT cache self-heal skipped: %r", exc)
        return []


def get_jit_vendor() -> str:
    return "hip" if torch.version.hip else "cuda"


def _jit_build_dir_name(module_name: str, build_hash: str = "") -> str:
    # Key on arch + tvm-ffi ABI too (module_name only hashes sources), so a
    # shared cache volume never reuses a cross-arch/ABI .so.
    #
    # The VENDOR belongs in the key as well: the arch tag is derived from
    # torch.cuda.get_device_capability(), and that namespace collides across
    # vendors -- an AMD gfx900 reports (9, 0), i.e. the same "9.0" as an NVIDIA
    # Hopper part. Without the vendor, a shared cache volume would hand a
    # gfx900 .so to an sm_90 rank (or the reverse) under an identical key,
    # which is exactly the cross-arch reuse this name exists to prevent.
    # Same defect and same fix shape as the HTCCL device-extension cache key.
    #
    # ``build_hash`` closes the last gap in the name: flags, wrappers and the
    # toolchain choice also decide the binary, and two builds that differ only
    # in those used to land in one directory. The .so file name is deliberately
    # left alone (it is module_name) so an operator can still recognise an
    # entry by kernel.
    vendor = get_jit_vendor()
    arch = get_jit_cuda_arch().target_name
    name = f"{module_name}__{vendor}_arch_{arch}__tvmffi_{_tvm_ffi_version()}"
    return f"{name}__b{build_hash}" if build_hash else name


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    extra_dependencies: List[str] | None = None,
    build_directory: str | None = None,
    header_only: bool = True,
) -> Module:
    """
    Loading a JIT module from C++/CUDA source files.
    We define a wrapper as a tuple of (export_name, kernel_name),
    where `export_name` is the name used to called from Python,
    and `kernel_name` is the name of the kernel class in C++/CUDA source.

    :param args: Unique marker of the JIT module. Must be distinct for different kernels.
    :type args: str
    :param cpp_files: A list of C++ source files.
    :type cpp_files: List[str] | None
    :param cuda_files: A list of CUDA source files.
    :type cuda_files: List[str] | None
    :param cpp_wrappers: A list of C++ wrappers, defining the export name and kernel name.
    :type cpp_wrappers: List[Tuple[str, str]] | None
    :param cuda_wrappers: A list of CUDA wrappers, defining the export name and kernel name.
    :type cuda_wrappers: List[Tuple[str, str]] | None
    :param extra_cflags: Extra C++ compiler flags.
    :type extra_cflags: List[str] | None
    :param extra_cuda_cflags: Extra CUDA compiler flags.
    :type extra_cuda_cflags: List[str] | None
    :param extra_ldflags: Extra linker flags.
    :type extra_ldflags: List[str] | None
    :param extra_include_paths: Extra include paths.
    :type extra_include_paths: List[str] | None
    :param extra_dependencies: Extra dependencies for the JIT module, e.g., cutlass.
    :type extra_dependencies: List[str] | None
    :param build_directory: The build directory for JIT compilation.
    :type build_directory: str | None
    :param header_only: Whether the module is header-only.
                        If true, apply the wrappers to export given class/functions.
                        Otherwise, we must export from C++/CUDA side.
    :return: A just-in-time(JIT) compiled module.
    :rtype: Module
    """

    from tvm_ffi.cpp import load, load_inline

    # Tell tvm_ffi which toolchain to use INSTEAD of letting it guess from the
    # filesystem.
    #
    # tvm_ffi._detect_gpu_backend() decides like this:
    #     try:  _find_rocm_home();  return "hip"
    #     except RuntimeError:      return "cuda"
    # and _find_rocm_home() falls back to "/opt/rocm" whenever that directory
    # merely EXISTS. So on any machine with ROCm installed it returns "hip" for
    # EVERY process -- including a CUDA one. Observed on the cross-vendor host:
    # a CUDA venv (torch.version.cuda=13.0, torch.version.hip=None) driving an
    # RTX 2080 Ti tried to build with
    #     /opt/rocm/bin/hipcc --offload-arch=gfx900 --offload-arch=gfx90c
    #     -DSGL_CUDA_ARCH=750
    # i.e. the AMD compiler, AMD arch flags and an NVIDIA arch macro at once.
    #
    # This is not exotic: a vendor-mixed host has BOTH runtimes installed by
    # construction, so filesystem probing is guaranteed to be wrong for one of
    # them. The process's own torch build is the only authority on which
    # toolchain it needs, and `backend=` is tvm_ffi's supported way to say so
    # (an explicit parameter, not an environment trick -- environment guessing
    # is what caused this).
    _backend = "hip" if is_hip_runtime() else "cuda"

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "csrc" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "csrc" / f).resolve()) for f in cuda_files]

    for dep in set(extra_dependencies or []):
        if dep not in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep} is not registered.")
        extra_include_paths += _REGISTERED_DEPENDENCIES[dep]()

    source_hash = ""
    module_name = "sgl_kernel_jit_" + "_".join(str(arg) for arg in args)
    if cpp_files or cuda_files:
        source_hash = _local_jit_source_hash(cpp_files + cuda_files)
        module_name += "_" + source_hash

    all_cflags = DEFAULT_CFLAGS + extra_cflags
    all_cuda_cflags = _get_default_target_flags() + extra_cuda_cflags
    all_ldflags = DEFAULT_LDFLAGS + extra_ldflags
    all_include_paths = DEFAULT_INCLUDE + extra_include_paths

    arch = get_jit_cuda_arch().target_name
    vendor = get_jit_vendor()
    build_hash = _build_input_hash(
        arch=arch,
        vendor=vendor,
        backend=_backend,
        header_only=header_only,
        cpp_wrappers=cpp_wrappers or [],
        cuda_wrappers=cuda_wrappers or [],
        extra_cflags=all_cflags,
        extra_cuda_cflags=all_cuda_cflags,
        extra_ldflags=all_ldflags,
        dependencies=extra_dependencies or [],
    )
    provenance = _provenance_record(
        module_name=module_name,
        source_hash=source_hash,
        build_hash=build_hash,
        arch=arch,
        vendor=vendor,
        include_paths=all_include_paths,
    )

    # A built .so under a deterministic dir is content-addressed: load it
    # directly to skip ninja, whose mtime check rebuilds every CI run (pip
    # install bumps dep header mtimes).
    caller_supplied_dir = build_directory is not None
    if build_directory is None:
        cache_dir = os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
        build_directory = str(
            pathlib.Path(cache_dir).expanduser()
            / _jit_build_dir_name(module_name, build_hash)
        )
    # Self-heal the cache before trusting anything in it. A build killed
    # mid-flight (the deadline collision of the sibling fix made that routine)
    # leaves build.ninja + cuda.cu + cuda_0.o.d and NO .so, and tvm-ffi hands
    # that wreck back forever: "Check failed: (lib_handle_ != nullptr)". Four
    # such directories once had to be removed by hand before this host would
    # boot. Complete entries -- anything with a .so -- are never touched.
    _selfheal_jit_cache_once()

    prebuilt = pathlib.Path(build_directory) / f"{module_name}.so"
    if prebuilt.is_file():
        # The directory name says what this entry SHOULD be; the provenance
        # record says what it IS. Only the second one is evidence.
        #
        # The cache root is one directory per host, shared by every checkout,
        # git worktree and wheel install on it, and the entry name carries no
        # tree identity -- by design, so a warm cache survives a move. The cost
        # of that design is that reuse across trees has to be checked, not
        # assumed. It was not: an sm_120 gptq_marlin_repack.so compiled in
        # /spinning/wt-merge-probe was loaded by a probe in a different
        # worktree and launched on an sm_86 card, which failed with "no kernel
        # image is available" pointing at a source file in a tree the running
        # process had never read.
        #
        # A caller-supplied directory is outside this scheme in both
        # directions: nothing stamps a record there, so nothing may demand one
        # -- verifying it would purge the caller's directory on every call.
        if caller_supplied_dir:
            ok, why = True, "caller-supplied build directory"
        elif not provenance_check_enabled():
            ok, why = True, "provenance check disabled"
        else:
            ok, why = check_provenance(build_directory, provenance)
        if not ok:
            logger.warning(
                "Discarding cached JIT module %s in %s: %s. It will be rebuilt "
                "from %s.",
                module_name,
                build_directory,
                why,
                KERNEL_PATH,
            )
            purge_entry(build_directory)
        else:
            from tvm_ffi import load_module

            try:
                module = load_module(str(prebuilt))
                logger.debug("Reused cached JIT module %s (%s)", module_name, why)
                return module
            except Exception:
                logger.warning(
                    "Cached JIT module %s failed to load; rebuilding.", module_name
                )
                # The .so is there and unusable -- truncated, cross-ABI, or
                # written by a build that died during the link. Rebuilding ON
                # TOP of it lets ninja's mtime check decide everything is up to
                # date and hand the same broken artefact back. Take the
                # directory with it.
                purge_entry(build_directory)
    else:
        # No artefact: if what is there is residue rather than a peer's live
        # build, discard it so ninja starts from a clean directory.
        heal_entry(build_directory)

    if header_only:
        cpp_wrappers = cpp_wrappers or []
        cuda_wrappers = cuda_wrappers or []
        cpp_sources = [f'#include "{path}"' for path in cpp_files]
        cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

        # include cuda files
        cuda_sources = [f'#include "{path}"' for path in cuda_files]
        cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]
        # building_marker records host+pid for the duration of the build. It is
        # what lets a co-located rank's sweep tell "a peer is compiling this
        # right now" from "somebody was killed compiling this", without a lock
        # and without deleting live work.
        with building_marker(build_directory), _jit_compile_context():
            module = load_inline(
                module_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                extra_cflags=all_cflags,
                extra_cuda_cflags=all_cuda_cflags,
                extra_ldflags=all_ldflags,
                extra_include_paths=all_include_paths,
                build_directory=build_directory,
                backend=_backend,
            )
    else:
        assert cpp_wrappers is None and cuda_wrappers is None
        with building_marker(build_directory), _jit_compile_context():
            module = load(
                module_name,
                cpp_files=cpp_files,
                cuda_files=cuda_files,
                extra_cflags=all_cflags,
                extra_cuda_cflags=all_cuda_cflags,
                extra_ldflags=all_ldflags,
                extra_include_paths=all_include_paths,
                backend=_backend,
                build_directory=build_directory,
            )

    # Stamped only after the build returned: a record next to a missing or
    # half-linked .so would vouch for an artefact that does not exist. A
    # caller-supplied directory is not ours to annotate -- it is not keyed by
    # this module's naming scheme, so a record there would be read back against
    # a name it never promised to match.
    if not caller_supplied_dir:
        _write_provenance(build_directory, provenance)
    return module


@dataclass
class ArchInfo:
    major: int
    minor: int
    suffix: str

    @property
    def target_name(self) -> str:
        return f"{self.major}.{self.minor}{self.suffix}"

    @property
    def jit_flag(self) -> str:
        return f"-DSGL_CUDA_ARCH={self.major * 100 + self.minor * 10}"


#: device index -> its architecture. Resolved once per DEVICE, not once per
#: process: a process that touches two cards of different generations must not
#: describe the second one with the first one's capability. See
#: `cache_once_per_arch` for what that cost when the memo was process-global,
#: and `srt/utils/cute_dsl_arch.py` for the same defect in the cutlass-DSL JIT.
_DEVICE_ARCH: Dict[int, ArchInfo] = {}

#: Set by `override_jit_cuda_arch` only. An explicit target beats the device.
_CUDA_ARCH: Optional[ArchInfo] = None


def _current_device_index() -> int:
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return -1


def _resolve_device_arch(index: int) -> ArchInfo:
    if index < 0:
        logger.warning("Cannot detect CUDA architecture.")
        return ArchInfo(0, 0, "")  # invalid value to trigger a compile error
    try:
        major, minor = torch.cuda.get_device_capability(index)
    except Exception:
        logger.warning("Cannot detect CUDA architecture of device %d.", index)
        return ArchInfo(0, 0, "")
    return ArchInfo(major, minor, "")


def get_jit_target_archs() -> List[str]:
    """Every architecture this process could dispatch a JIT kernel to.

    The union over the VISIBLE devices. Under the rank-isolation the fork uses
    (one physical GPU per worker via CUDA_VISIBLE_DEVICES) that is a single
    entry and matches the build target exactly. It is longer only in a process
    that deliberately drives a whole rig at once -- the card probe, the planner
    sweep -- which is precisely the shape that has to build one module per
    architecture rather than one per process.
    """
    archs: List[str] = []
    try:
        count = torch.cuda.device_count()
    except Exception:
        return archs
    for index in range(count):
        name = _resolve_device_arch(index).target_name
        if name not in archs:
            archs.append(name)
    return sorted(archs)


@contextmanager
def _jit_compile_context():
    if is_hip_runtime():
        yield  # TODO: support ROCm `TVM_FFI_ROCM_ARCH_LIST` if needed
        return
    env_key = "TVM_FFI_CUDA_ARCH_LIST"
    old_value = os.environ.get(env_key, None)
    os.environ[env_key] = get_jit_cuda_arch().target_name
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_value


# NOTE: this might also be used in __main__.py for compile flags export
def _get_default_target_flags() -> List[str]:
    if is_hip_runtime():
        flags = ["-DUSE_ROCM", "-std=c++20", "-O3"]
        # Detect FP8 type based on GPU architecture
        try:
            device = torch.cuda.current_device()
            gcn_arch = torch.cuda.get_device_properties(device).gcnArchName
            if "gfx942" in gcn_arch:
                flags.append("-DHIP_FP8_TYPE_FNUZ=1")
            else:
                flags.append("-DHIP_FP8_TYPE_E4M3=1")
        except Exception:
            flags.append("-DHIP_FP8_TYPE_E4M3=1")
        return flags
    else:
        return [
            get_jit_cuda_arch().jit_flag,
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
        ]


@contextmanager
def override_jit_cuda_arch(major: int, minor: int, suffix: str = ""):
    """A context manager to temporarily override CUDA architecture."""
    global _CUDA_ARCH
    old_value = _CUDA_ARCH
    _CUDA_ARCH = ArchInfo(major, minor, suffix)
    try:
        yield
    finally:
        _CUDA_ARCH = old_value


def get_jit_cuda_arch() -> ArchInfo:
    """The architecture JIT code must be built for right now.

    An explicit override wins; otherwise it is the capability of the CURRENT
    device. Reading the current device on every call rather than once per
    process is the whole point: the answer is a per-device property, and a
    process is free to switch cards between two calls.
    """
    if _CUDA_ARCH is not None:
        return _CUDA_ARCH
    index = _current_device_index()
    arch = _DEVICE_ARCH.get(index)
    if arch is None:
        arch = _resolve_device_arch(index)
        if arch.major or arch.minor:
            # Never memoize the failure sentinel: a probe that ran before the
            # runtime was up would otherwise pin (0, 0) for the rest of the
            # process and turn a transient miss into a permanent one.
            _DEVICE_ARCH[index] = arch
    return arch


@cache_once_per_arch
def is_arch_support_pdl() -> bool:
    if is_hip_runtime() or is_musa_runtime():
        return False
    return get_jit_cuda_arch().major >= 9


def _find_package_root(package: str) -> Optional[pathlib.Path]:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).resolve().parent


# NOTE: this might also be used in __main__.py for compile flags export
_REGISTERED_DEPENDENCIES: Dict[str, Callable[[], List[str]]] = {}


def register_dependency(name: str):
    def decorator(f: Callable[[], List[str]]) -> Callable[[], List[str]]:
        if name in _REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {name} already registered")
        _REGISTERED_DEPENDENCIES[name] = f
        return f

    return decorator


@register_dependency("flashinfer")
def get_flashinfer_include_paths() -> List[str]:
    include_paths: List[str] = []
    flashinfer_root = _find_package_root("flashinfer")
    if flashinfer_root is None:
        raise RuntimeError(
            "Cannot find flashinfer package. Please install flashinfer to get"
            "the required headers for JIT compilation."
        )

    flashinfer_data = flashinfer_root / "data"
    candidates = [
        flashinfer_data / "include",
        flashinfer_data / "csrc",
        flashinfer_data / "cutlass" / "include",
        flashinfer_data / "cutlass" / "tools" / "util" / "include",
        flashinfer_data / "spdlog" / "include",
    ]

    for path in candidates:
        if not path.exists():
            raise RuntimeError(
                f"Required header path {path} for flashinfer dependency not found."
                " Please check your flashinfer installation."
            )
        include_paths.append(str(path))
    return include_paths


def get_mathdx_root() -> Optional[pathlib.Path]:
    """Locate the NVIDIA Math-DX install (cuBLASDx headers).

    Searches in order:
      1. ``$MATHDX_HOME`` env var (extracted Math-DX archive root).
      2. The ``nvidia-mathdx`` PyPI package, if installed.
    """
    env_home = os.environ.get("MATHDX_HOME")
    if env_home:
        candidate = pathlib.Path(env_home).expanduser().resolve()
        if (candidate / "include").exists():
            return candidate

    # The ``nvidia-mathdx`` wheel installs as the namespace package
    # ``nvidia.mathdx`` (no __init__, so spec.origin is None); resolve it via
    # submodule_search_locations rather than _find_package_root, which only
    # handles regular packages.
    spec = importlib.util.find_spec("nvidia.mathdx")
    if spec is not None:
        roots = list(spec.submodule_search_locations or [])
        if spec.origin is not None:
            roots.append(str(pathlib.Path(spec.origin).parent))
        for root in roots:
            candidate = pathlib.Path(root).resolve()
            if (candidate / "include").exists():
                return candidate

    return None


@register_dependency("mathdx")
def get_mathdx_include_paths() -> List[str]:
    root = get_mathdx_root()
    if root is None:
        raise RuntimeError(
            "Cannot find NVIDIA Math-DX (cuBLASDx) headers. "
            "Install the `nvidia-mathdx` package "
            "(`pip install nvidia-mathdx`) or set MATHDX_HOME to an "
            "extracted Math-DX archive root."
        )
    candidates = [root / "include"]
    cutlass = root / "external" / "cutlass" / "include"
    if cutlass.exists():
        candidates.append(cutlass)
    return [str(p) for p in candidates]


@register_dependency("cutlass")
def get_cutlass_include_paths() -> List[str]:
    include_paths: List[str] = []

    flashinfer_root = _find_package_root("flashinfer")
    if flashinfer_root is not None:
        candidates = [
            flashinfer_root / "data" / "cutlass" / "include",
            flashinfer_root / "data" / "cutlass" / "tools" / "util" / "include",
        ]
        for path in candidates:
            if path.exists():
                include_paths.append(str(path))

    deep_gemm_root = _find_package_root("deep_gemm")
    if deep_gemm_root is not None:
        candidate = deep_gemm_root / "include"
        if candidate.exists():
            include_paths.append(str(candidate))

    # De-duplicate while preserving order.
    unique_paths = []
    seen = set()
    for path in include_paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    if not unique_paths:
        raise RuntimeError(
            "Cannot find CUTLASS headers required for JIT compilation. "
            "Please install flashinfer or deep_gemm with CUTLASS headers."
        )
    return unique_paths


__all__ = [
    "should_run_full_tests",
    "get_ci_test_range",
    "cache_once",
    "cache_once_per_arch",
    "check_provenance",
    "is_hip_runtime",
    "make_cpp_args",
    "load_jit",
    "override_jit_cuda_arch",
    "get_jit_cuda_arch",
    "get_jit_target_archs",
    "get_jit_vendor",
    "is_arch_support_pdl",
    "register_dependency",
]
