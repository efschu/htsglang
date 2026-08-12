"""#384: the sgl_kernel wheel-shadow detector, proven on fabricated layouts.

The defect being guarded is silent by construction. Two distributions provide
the same ``sgl_kernel`` import package -- pypi's armless ``sgl-kernel`` and the
fork's ``sglang-kernel``, which carries ``int8_scaled_mm`` -- and because their
distribution NAMES differ, pip sees no conflict. Whichever was installed last
owns the files. Nothing fails at install time, and the loss surfaces much later
during layer construction.

So the detector has to be exercised against layouts that do not exist on this
machine, which is what these tests build: real ``.dist-info`` directories with
real RECORD files, and real ELF objects, in ``tmp_path``.

WHY THE ELF OBJECTS ARE SYNTHESISED RATHER THAN COPIED
------------------------------------------------------
The CUDA-major check (#436) reads ``DT_NEEDED``. Copying the installed 11 MB
object would make the test depend on the rig's current install -- the very
thing under test -- and would not let us produce the *bad* case at all, since
no cu12 wheel is installed here any more. ``_elf`` below writes a minimal but
genuine ELF64 shared object with a real dynamic section, so the parser is
tested against the format rather than against a mock of itself.

THE CASE THAT MATTERS MOST
--------------------------
``test_shadow_trips_even_while_the_arm_is_still_present``. A venv where BOTH
dists are installed and the fork's files currently win looks perfect from
every angle an import can see: right version, right arm, right tree. It is
also one unrelated ``pip install`` from serving an armless kernel. A detector
that only reports the already-broken state would pass that venv, and the rig
has been through the resulting failure twice (#357).
"""

import struct
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.srt.utils import kernel_dist_guard as G
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GUARD_SOURCE = Path(G.__file__)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _elf(path: Path, needed=("libcudart.so.13", "libc.so.6"), extra=b"") -> Path:
    """Write a minimal, genuine ELF64 shared object carrying DT_NEEDED.

    Layout: header, two program headers (PT_LOAD covering the whole file with
    an identity vaddr mapping, plus PT_DYNAMIC), then the string table, then
    the dynamic array. ``extra`` is appended so a test can plant (or withhold)
    the INT8 arm symbol the detector scans for.
    """
    ehsize, phentsize, phnum = 64, 56, 2
    phoff = ehsize
    strtab_off = phoff + phentsize * phnum

    strtab = b"\0"
    offsets = []
    for name in needed:
        offsets.append(len(strtab))
        strtab += name.encode() + b"\0"

    dyn_off = strtab_off + len(strtab)
    dyn = b""
    for off in offsets:
        dyn += struct.pack("<qQ", 1, off)  # DT_NEEDED
    dyn += struct.pack("<qQ", 5, strtab_off)  # DT_STRTAB (identity-mapped)
    dyn += struct.pack("<qQ", 0, 0)  # DT_NULL
    total = dyn_off + len(dyn) + len(extra)

    eh = bytearray(64)
    eh[0:4] = b"\x7fELF"
    eh[4] = 2  # ELFCLASS64
    eh[5] = 1  # ELFDATA2LSB
    eh[6] = 1  # EV_CURRENT
    struct.pack_into("<H", eh, 0x10, 3)  # e_type = ET_DYN
    struct.pack_into("<H", eh, 0x12, 0x3E)  # e_machine = x86-64
    struct.pack_into("<I", eh, 0x14, 1)  # e_version
    struct.pack_into("<Q", eh, 0x20, phoff)
    struct.pack_into("<H", eh, 0x34, ehsize)
    struct.pack_into("<H", eh, 0x36, phentsize)
    struct.pack_into("<H", eh, 0x38, phnum)

    # PT_LOAD over the whole file, vaddr == offset so DT_STRTAB resolves.
    ph = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, total, total, 0x1000)
    ph += struct.pack(
        "<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off, len(dyn), len(dyn), 8
    )

    blob = bytes(eh) + ph
    blob += b"\0" * (strtab_off - len(blob))
    blob += strtab + dyn + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def _dist_info(
    sp: Path,
    dist: str,
    version: str,
    n_files: int,
    *,
    sha256: str = None,
    url: str = None,
) -> Path:
    """Create a .dist-info claiming ``n_files`` files under ``sgl_kernel/``."""
    di = sp / f"{dist.replace('-', '_')}-{version}.dist-info"
    di.mkdir(parents=True)
    (di / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n\nbody\n"
    )
    rows = [f"sgl_kernel/f{i}.py,sha256=x,1" for i in range(n_files)]
    rows.append(f"{di.name}/METADATA,sha256=y,2")
    (di / "RECORD").write_text("\n".join(rows) + "\n")
    if sha256:
        (di / "direct_url.json").write_text(
            '{"archive_info": {"hashes": {"sha256": "%s"}}, "url": "%s"}'
            % (sha256, url or "file:///tmp/w.whl")
        )
    return di


def _package(sp: Path, *, arm: bool, cudart: str = "libcudart.so.13") -> Path:
    """Create the sgl_kernel/ package dir with one object in it."""
    pkg = sp / "sgl_kernel"
    _elf(
        pkg / "sm100" / "common_ops.abi3.so",
        needed=(cudart, "libcublas.so." + cudart.rsplit(".", 1)[1]),
        extra=b"\0schema:int8_scaled_mm(...)\0" if arm else b"\0schema:gemm\0",
    )
    return pkg


def _torch(sp: Path, cudart: str = "libcudart.so.13") -> None:
    _elf(sp / "torch" / "lib" / "libtorch_cuda.so", needed=(cudart,))


@pytest.fixture
def good(tmp_path):
    """The state the rig is actually in: one fork dist, armed, cu13."""
    sp = tmp_path / "site-packages"
    sp.mkdir()
    _dist_info(sp, "sglang-kernel", "0.4.4", 74, sha256=G.PINNED_WHEEL_SHA256)
    _package(sp, arm=True)
    _torch(sp)
    return sp


# --------------------------------------------------------------------------
# The ELF parser, against the format
# --------------------------------------------------------------------------


def test_elf_needed_reads_the_dynamic_section(tmp_path):
    so = _elf(tmp_path / "x.so", needed=("libcudart.so.13", "libcublasLt.so.13"))
    assert G.elf_needed(so) == ["libcudart.so.13", "libcublasLt.so.13"]
    assert G.cuda_majors(G.elf_needed(so)) == {"libcudart": 13, "libcublasLt": 13}


def test_elf_needed_refuses_a_non_elf(tmp_path):
    junk = tmp_path / "not.so"
    junk.write_bytes(b"MZ" + b"\0" * 200)
    with pytest.raises(G.ElfError):
        G.elf_needed(junk)


def test_cuda_majors_ignores_unversioned_and_foreign_sonames():
    """A substring search would have called this a cu12 install."""
    assert G.cuda_majors(["libc.so.6", "libcudart.so", "libfoo_cudart.so.12"]) == {}


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


def test_clean_fork_install_is_armed(good):
    r = G.inspect_sgl_kernel(good)
    assert r.verdict == G.VERDICT_ARMED
    assert r.ok and r.arm_present and not r.cuda_major_split
    assert r.sha256_mismatch is None
    assert G.describe_problems(r, require_arm=True) == []


def test_the_pypi_shadow_trips_the_detector(good):
    """RED CASE: the armless pypi dist installed alongside the fork's."""
    _dist_info(good, "sgl-kernel", "0.3.21", 69)
    r = G.inspect_sgl_kernel(good, expect_sha256=None)
    assert r.verdict == G.VERDICT_SHADOWED
    assert not r.ok
    problems = G.describe_problems(r, require_arm=False)
    assert any("WHEEL SHADOW" in p for p in problems), problems
    assert {p.dist_name for p in r.providers} == {"sgl-kernel", "sglang-kernel"}


def test_shadow_trips_even_while_the_arm_is_still_present(good):
    """The state an import-based check cannot see.

    Fork files win, arm present, version right -- and two dists installed. The
    detector must refuse anyway: nothing about this venv is stable.
    """
    _dist_info(good, "sgl-kernel", "0.3.21", 69)
    r = G.inspect_sgl_kernel(good, expect_sha256=None)
    assert r.arm_present, "precondition: the good files still win"
    assert r.verdict == G.VERDICT_SHADOWED
    assert not r.ok
    assert any(
        "currently" in p and "present" in p
        for p in G.describe_problems(r, require_arm=True)
    )


def test_armless_single_dist_is_a_verdict_not_an_error(tmp_path):
    """A stock pypi install is a supported image, so policy decides."""
    sp = tmp_path / "site-packages"
    sp.mkdir()
    _dist_info(sp, "sgl-kernel", "0.3.21", 69)
    _package(sp, arm=False)
    _torch(sp)
    r = G.inspect_sgl_kernel(sp, expect_sha256=None)
    assert r.verdict == G.VERDICT_ARMLESS
    assert r.ok, "single-dist armless is structurally fine"
    assert G.describe_problems(r, require_arm=False) == []
    assert G.describe_problems(r, require_arm=True), "but not when the arm is required"


def test_cuda_major_split_is_detected(tmp_path):
    """#436: wheel links cu12 while torch links cu13 -> deterministic SIGSEGV."""
    sp = tmp_path / "site-packages"
    sp.mkdir()
    _dist_info(sp, "sglang-kernel", "0.4.4", 74, sha256=G.PINNED_WHEEL_SHA256)
    _package(sp, arm=True, cudart="libcudart.so.12")
    _torch(sp, cudart="libcudart.so.13")
    r = G.inspect_sgl_kernel(sp)
    assert r.cuda_major_split
    assert not r.ok
    assert any("#436" in p for p in G.describe_problems(r, require_arm=True))


def test_off_pin_wheel_is_reported(good):
    G_sp = good
    for di in G_sp.glob("*.dist-info"):
        (di / "direct_url.json").write_text(
            '{"archive_info": {"hashes": {"sha256": "dead"}}, "url": "file:///w"}'
        )
    r = G.inspect_sgl_kernel(G_sp)
    assert r.sha256_mismatch and "dead" in r.sha256_mismatch
    assert not r.ok


def test_index_install_counts_as_off_pin(good):
    """No direct_url.json at all means it came from an index, not the pin."""
    for di in good.glob("*.dist-info"):
        (di / "direct_url.json").unlink()
    r = G.inspect_sgl_kernel(good)
    assert r.sha256_mismatch and "direct_url" in r.sha256_mismatch


def test_missing_install(tmp_path):
    sp = tmp_path / "site-packages"
    sp.mkdir()
    r = G.inspect_sgl_kernel(sp, expect_sha256=None)
    assert r.verdict == G.VERDICT_MISSING
    assert not r.ok


def test_list_providers_is_generic_over_the_package_name(good):
    assert [p.dist_name for p in G.list_providers("sgl_kernel", good)] == [
        "sglang-kernel"
    ]
    assert G.list_providers("nothing_here", good) == []


# --------------------------------------------------------------------------
# The properties the Docker build layer depends on
# --------------------------------------------------------------------------


def test_the_guard_imports_only_the_standard_library():
    """It must run in a build layer where torch and sglang are not importable.

    Enforced structurally rather than by comment: a future edit that reaches
    for ``torch`` or a fork helper would break the Docker assert at image
    build time, which is the worst place to find out.
    """
    import ast

    tree = ast.parse(GUARD_SOURCE.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    assert modules <= set(sys.stdlib_module_names), modules - set(
        sys.stdlib_module_names
    )


def test_cli_runs_by_path_without_importing_sglang(good, tmp_path):
    """Exactly how the Dockerfile invokes it: a plain path, no PYTHONPATH."""
    proc = subprocess.run(
        [
            sys.executable,
            "-I",  # isolated: ignore PYTHONPATH and user site entirely
            str(GUARD_SOURCE),
            "--site-packages",
            str(good),
            "--require-arm",
            "--expect-pinned-sha256",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "verdict=ARMED" in proc.stdout


def test_cli_exits_nonzero_and_loudly_on_the_shadow(good, tmp_path):
    _dist_info(good, "sgl-kernel", "0.3.21", 69)
    proc = subprocess.run(
        [sys.executable, "-I", str(GUARD_SOURCE), "--site-packages", str(good)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 1
    # Loud: the verdict, both dists, and the repair recipe, without being asked.
    assert "SHADOWED" in proc.stdout
    assert "sgl-kernel" in proc.stdout and "sglang-kernel" in proc.stdout
    assert "pip uninstall -y sgl-kernel" in proc.stdout


def test_cli_wheel_mode_refuses_an_off_pin_build_input(tmp_path):
    wheel = tmp_path / "sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl"
    wheel.write_bytes(b"not the pinned wheel")
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(GUARD_SOURCE),
            "--wheel",
            str(wheel),
            "--expect-pinned-sha256",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "does not match the pin" in proc.stderr
    assert G.sha256_file(wheel) in proc.stdout


def test_cli_wheel_mode_accepts_a_matching_hash(tmp_path):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"payload")
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(GUARD_SOURCE),
            "--wheel",
            str(wheel),
            "--expect-sha256",
            G.sha256_file(wheel),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
