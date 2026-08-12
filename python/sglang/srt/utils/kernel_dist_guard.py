# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Detector for the #384 ``sgl_kernel`` wheel-shadow state.

WHAT THIS GUARDS
----------------
Two different *distributions* ship the same *import package* ``sgl_kernel``:

===========================  ==================  =================
dist name                    origin              INT8 arm
===========================  ==================  =================
``sgl-kernel``               pypi                no
``sglang-kernel``            this fork's build   yes
===========================  ==================  =================

pip sees no conflict, because the distribution names differ -- so whichever was
installed *last* owns the files on disk. Any ``pip install`` / ``pip install -U``
/ requirements sync that touches ``sgl-kernel`` restores the armless files over
the fork's and silently removes ``int8_scaled_mm``, which the INT8-W8A8
production default needs. The failure then surfaces far from its cause, during
layer construction inside the JIT cold-build window.

Reference: ``docs/rig-runbook.md`` section 2.1 (provenance, pin, both-directions
verify, and the repair recipe), and the #357 roll-forward/roll-back pair, which
flipped the same files twice.

WHY THIS MODULE DOES NOT ``import sgl_kernel``
----------------------------------------------
Two independent reasons, both load-bearing:

1. **It must run where an import cannot.** The primary caller is a Docker
   build-time RUN layer. ``docker/htsglang.Dockerfile`` (the build-time verify
   step) records that a real ``import torch`` / ``import sgl_kernel`` needs
   ``libcuda.so.1`` from the host driver, which is absent during ``docker
   build``. An import-based assert would fail on a *correct* image.
2. **Importing is itself the hazard.** The runbook's own installed-wheel
   verification is deliberately all file inspection, because ``import
   sgl_kernel`` on a busy shared box is exactly what section 2.1 warns about.

So every check here is filesystem + packaging metadata + ELF inspection, and the
module imports nothing outside the standard library. That also lets it be run by
plain path (``python3 .../kernel_dist_guard.py``) without importing the
``sglang`` package, whose ``srt.utils.__init__`` pulls in torch.

VERDICTS
--------
``ARMED``
    Exactly one distribution provides ``sgl_kernel`` and the INT8 arm is
    present. The intended state of the rig venv and of an INT8-capable image.
``ARMLESS``
    Exactly one distribution provides ``sgl_kernel`` and the INT8 arm is
    absent. This is a *legitimate* state -- the stock pypi wheel, e.g. a slim
    image built with ``INSTALL_SGL_KERNEL=0`` semantics or a Turing-only image
    -- so it is only an error when the caller passes ``require_arm``.
``SHADOWED``
    More than one distribution provides ``sgl_kernel``. This is the #384 hazard
    itself and is ALWAYS an error, even when the arm currently happens to be
    present: the next ``pip install`` decides which files win, so a shadowed
    venv is one unrelated pip invocation away from a silent regression.
``MISSING``
    No distribution provides ``sgl_kernel``.

The distinction between ``ARMLESS`` and ``SHADOWED`` is the point of the module.
An armless image can be a deliberate build choice; a shadowed one never is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "KernelDistReport",
    "ProviderDist",
    "VERDICT_ARMED",
    "VERDICT_ARMLESS",
    "VERDICT_MISSING",
    "VERDICT_SHADOWED",
    "PINNED_WHEEL_SHA256",
    "inspect_sgl_kernel",
    "list_providers",
    "elf_needed",
    "main",
]

VERDICT_ARMED = "ARMED"
VERDICT_ARMLESS = "ARMLESS"
VERDICT_SHADOWED = "SHADOWED"
VERDICT_MISSING = "MISSING"

#: The import package both distributions provide -- the whole reason #384 exists.
IMPORT_PACKAGE = "sgl_kernel"

#: The symbol whose absence breaks the INT8-W8A8 production default. Searched
#: for as a byte string inside the compiled objects: the shipped ``.so`` files
#: are stripped, so ``nm`` shows nothing and the runbook uses ``strings`` for
#: exactly this reason (section 2.1, the #398 install-verification table).
INT8_ARM_SYMBOL = b"int8_scaled_mm"

#: Provenance pin from ``docs/rig-runbook.md`` section 2.1. Kept here as a
#: default so a caller can verify the *installed* dist against the documented
#: wheel without re-typing the hash; ``--expect-sha256`` overrides it, and
#: passing ``expect_sha256=None`` skips the comparison entirely.
PINNED_WHEEL_SHA256 = "67f03cfa755efa01498c7732bd6ae015ec5673feffe9a51452fefdbe0dcd4664"

#: Repair recipe, quoted in every loud failure so the reader never has to go
#: looking for it. Mirrors "Making it durable" in runbook section 2.1.
REPAIR_RECIPE = """\
Repair (run only when the venv is QUIET -- check that no live process maps the
files first: grep -c sgl_kernel /proc/<pid>/maps over every process on this
interpreter; removing them under a running server is how a working rig becomes a
broken one mid-measurement):

    V={venv}
    $V/bin/pip uninstall -y sgl-kernel          # drop the shadowing pypi dist
    $V/bin/pip install --no-deps --force-reinstall <fork wheel>.whl

Then verify BOTH directions, as #357 did -- the fork version reported AND the
arm present. A version bump alone is not evidence. See docs/rig-runbook.md 2.1.\
"""


# ---------------------------------------------------------------------------
# ELF inspection (the #436 trap)
# ---------------------------------------------------------------------------
#
# The wheel and torch must link the SAME CUDA major. When they diverge,
# `cudaRuntimeGetVersion` (a linked, version-tagged import that binds to the
# cudart the wheel was built against) and `dlsym(RTLD_DEFAULT, ...)` (unversioned,
# whole-process, resolved by load order -> torch's cudart) disagree about the
# ABI of `cudaMemcpyBatchAsync`, and the 9-argument convention gets applied to
# the 8-argument function. That is a deterministic SIGSEGV, not a race; see
# runbook section 2.1, "cu13 rebuild (#436)".
#
# The DT_NEEDED entries are parsed out of the ELF properly rather than grepped
# for, because a substring search for b"libcudart.so.12" also hits unrelated
# strings (log text, embedded paths, another library's soname) and would report
# a split that is not there.

_PT_LOAD = 1
_PT_DYNAMIC = 2
_DT_NULL = 0
_DT_NEEDED = 1
_DT_STRTAB = 5

_CUDA_SONAME_RE = re.compile(r"^lib(cudart|cublas|cublasLt|nvrtc)\.so\.(\d+)$")


class ElfError(RuntimeError):
    """Raised when a file is not a 64-bit little-endian ELF we can read."""


def elf_needed(path: Path) -> List[str]:
    """Return the ``DT_NEEDED`` sonames of a 64-bit little-endian ELF object.

    A minimal, dependency-free equivalent of ``objdump -p <so> | grep NEEDED``,
    which is the command the runbook prescribes for this check. Only the ELF
    class this project ships (ELF64, little endian) is supported; anything else
    raises :class:`ElfError` rather than guessing.
    """
    data = path.read_bytes()
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise ElfError(f"{path}: not an ELF file")
    if data[4] != 2:  # EI_CLASS: 2 == ELFCLASS64
        raise ElfError(f"{path}: not ELF64")
    if data[5] != 1:  # EI_DATA: 1 == ELFDATA2LSB
        raise ElfError(f"{path}: not little-endian ELF")

    e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum = struct.unpack_from("<H", data, 0x38)[0]
    if e_phoff == 0 or e_phnum == 0:
        raise ElfError(f"{path}: no program headers")

    loads: List[Tuple[int, int, int]] = []  # (vaddr, filesz, offset)
    dynamic: Optional[Tuple[int, int]] = None  # (offset, filesz)
    for i in range(e_phnum):
        base = e_phoff + i * e_phentsize
        if base + 56 > len(data):
            raise ElfError(f"{path}: truncated program header table")
        p_type = struct.unpack_from("<I", data, base)[0]
        p_offset = struct.unpack_from("<Q", data, base + 0x08)[0]
        p_vaddr = struct.unpack_from("<Q", data, base + 0x10)[0]
        p_filesz = struct.unpack_from("<Q", data, base + 0x20)[0]
        if p_type == _PT_LOAD:
            loads.append((p_vaddr, p_filesz, p_offset))
        elif p_type == _PT_DYNAMIC:
            dynamic = (p_offset, p_filesz)

    if dynamic is None:
        return []

    def vaddr_to_offset(vaddr: int) -> Optional[int]:
        for p_vaddr, p_filesz, p_offset in loads:
            if p_vaddr <= vaddr < p_vaddr + p_filesz:
                return p_offset + (vaddr - p_vaddr)
        return None

    dyn_off, dyn_size = dynamic
    needed_offsets: List[int] = []
    strtab_vaddr: Optional[int] = None
    for pos in range(dyn_off, min(dyn_off + dyn_size, len(data) - 15), 16):
        d_tag, d_un = struct.unpack_from("<qQ", data, pos)
        if d_tag == _DT_NULL:
            break
        if d_tag == _DT_NEEDED:
            needed_offsets.append(d_un)
        elif d_tag == _DT_STRTAB:
            strtab_vaddr = d_un

    if strtab_vaddr is None or not needed_offsets:
        return []
    strtab_off = vaddr_to_offset(strtab_vaddr)
    if strtab_off is None:
        raise ElfError(f"{path}: DT_STRTAB {strtab_vaddr:#x} outside every PT_LOAD")

    out: List[str] = []
    for rel in needed_offsets:
        start = strtab_off + rel
        end = data.find(b"\0", start)
        if start >= len(data) or end < 0:
            raise ElfError(f"{path}: DT_NEEDED string offset {rel} out of range")
        out.append(data[start:end].decode("utf-8", "replace"))
    return out


def cuda_majors(sonames: Iterable[str]) -> Dict[str, int]:
    """Map ``{libcudart: 13, libcublas: 13, ...}`` from a soname list."""
    found: Dict[str, int] = {}
    for name in sonames:
        m = _CUDA_SONAME_RE.match(name)
        if m:
            found[f"lib{m.group(1)}"] = int(m.group(2))
    return found


# ---------------------------------------------------------------------------
# Packaging metadata
# ---------------------------------------------------------------------------


@dataclass
class ProviderDist:
    """One installed distribution that provides the ``sgl_kernel`` package."""

    dist_name: str
    version: str
    dist_info: Path
    #: Number of ``sgl_kernel/`` files this dist claims in its RECORD.
    recorded_files: int
    #: ``direct_url.json`` archive hash, when the dist was installed from a
    #: local wheel. ``None`` for index installs -- which is itself informative:
    #: the fork wheel is always a file:// install.
    direct_url_sha256: Optional[str] = None
    direct_url: Optional[str] = None

    @property
    def is_pypi_shadow(self) -> bool:
        """True for the armless pypi distribution named in runbook 2.1."""
        return self.dist_name == "sgl-kernel"


@dataclass
class KernelDistReport:
    """The full picture, so callers never have to re-inspect the filesystem."""

    verdict: str
    site_packages: Path
    providers: List[ProviderDist] = field(default_factory=list)
    package_dir: Optional[Path] = None
    arm_present: bool = False
    #: Objects scanned for the INT8 arm, and how many hits each had.
    arm_scan: Dict[str, int] = field(default_factory=dict)
    #: ``{libcudart: 13, ...}`` observed in the installed objects.
    kernel_cuda_majors: Dict[str, int] = field(default_factory=dict)
    #: The same, for torch -- ``None`` when torch is not installed here.
    torch_cuda_major: Optional[int] = None
    #: Set when the two disagree: the #436 trap.
    cuda_major_split: bool = False
    #: Set when a sha256 pin was requested and the installed dist does not match.
    sha256_mismatch: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only for a single-provider, cu-major-consistent, pinned install.

        Deliberately does NOT consider ``arm_present``: whether an armless
        install is acceptable is the caller's policy, expressed via
        ``require_arm``. Everything this property does cover is unconditionally
        wrong.
        """
        return (
            self.verdict in (VERDICT_ARMED, VERDICT_ARMLESS)
            and not self.cuda_major_split
            and self.sha256_mismatch is None
        )

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "site_packages": str(self.site_packages),
            "providers": [
                {
                    "dist_name": p.dist_name,
                    "version": p.version,
                    "dist_info": str(p.dist_info),
                    "recorded_files": p.recorded_files,
                    "direct_url": p.direct_url,
                    "direct_url_sha256": p.direct_url_sha256,
                }
                for p in self.providers
            ],
            "package_dir": str(self.package_dir) if self.package_dir else None,
            "arm_present": self.arm_present,
            "arm_scan": self.arm_scan,
            "kernel_cuda_majors": self.kernel_cuda_majors,
            "torch_cuda_major": self.torch_cuda_major,
            "cuda_major_split": self.cuda_major_split,
            "sha256_mismatch": self.sha256_mismatch,
            "ok": self.ok,
            "notes": self.notes,
        }


def _read_metadata(dist_info: Path) -> Tuple[str, str]:
    """Return ``(dist_name, version)`` from a ``.dist-info`` directory.

    Prefers the METADATA fields; falls back to the directory name, which
    encodes ``<name>-<version>.dist-info`` with the name normalised.
    """
    name = version = ""
    meta = dist_info / "METADATA"
    if meta.is_file():
        for line in meta.read_text("utf-8", "replace").splitlines():
            if not line.strip():
                break  # end of headers; the body is the long description
            low = line.lower()
            if low.startswith("name:") and not name:
                name = line.split(":", 1)[1].strip()
            elif low.startswith("version:") and not version:
                version = line.split(":", 1)[1].strip()
            if name and version:
                break
    if not name or not version:
        stem = dist_info.name[: -len(".dist-info")]
        guess_name, _, guess_version = stem.rpartition("-")
        name = name or guess_name.replace("_", "-")
        version = version or guess_version
    return name, version


def _record_provides_package(dist_info: Path, package: str = IMPORT_PACKAGE) -> int:
    """Count the ``<package>/`` entries in a dist's RECORD."""
    record = dist_info / "RECORD"
    if not record.is_file():
        return 0
    prefix = package + "/"
    count = 0
    for line in record.read_text("utf-8", "replace").splitlines():
        path = line.split(",", 1)[0].strip()
        if path.startswith(prefix):
            count += 1
    return count


def _read_direct_url(dist_info: Path) -> Tuple[Optional[str], Optional[str]]:
    du = dist_info / "direct_url.json"
    if not du.is_file():
        return None, None
    try:
        payload = json.loads(du.read_text("utf-8", "replace"))
    except json.JSONDecodeError:
        return None, None
    url = payload.get("url")
    archive = payload.get("archive_info") or {}
    sha = None
    hashes = archive.get("hashes") or {}
    if isinstance(hashes, dict) and hashes.get("sha256"):
        sha = hashes["sha256"]
    elif isinstance(archive.get("hash"), str) and archive["hash"].startswith("sha256="):
        sha = archive["hash"].split("=", 1)[1]
    return url, sha


def _detect_torch_cuda_major(site_packages: Path) -> Tuple[Optional[int], List[str]]:
    """Torch's CUDA major, by file inspection only (never ``import torch``).

    Primary source is the DT_NEEDED list of a torch CUDA object; the
    ``nvidia/cu<major>`` layout is used as a fallback because the venv this
    guards resolves nvcc and the cu13 runtime from exactly that directory
    (runbook 2.1, "There is no local CUDA 13 system toolkit").
    """
    notes: List[str] = []
    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.is_dir():
        for candidate in ("libtorch_cuda.so", "libtorch_global_deps.so"):
            so = torch_lib / candidate
            if not so.is_file():
                continue
            try:
                majors = cuda_majors(elf_needed(so))
            except (ElfError, OSError) as exc:
                notes.append(f"torch ELF probe failed on {candidate}: {exc}")
                continue
            if "libcudart" in majors:
                return majors["libcudart"], notes
            notes.append(f"{candidate} declares no versioned libcudart NEEDED")
    for child in sorted((site_packages / "nvidia").glob("cu*")):
        m = re.match(r"^cu(\d+)$", child.name)
        if m and child.is_dir():
            notes.append(f"torch CUDA major inferred from nvidia/{child.name} layout")
            return int(m.group(1)), notes
    notes.append("torch CUDA major not determinable by file inspection")
    return None, notes


def list_providers(
    package: str = IMPORT_PACKAGE, site_packages: Optional[Path] = None
) -> List[ProviderDist]:
    """Every installed distribution that ships files under ``<package>/``.

    The primitive the shadow check is built on, kept separate from
    :func:`inspect_sgl_kernel` because "how many distributions claim this
    import name" is a general packaging question, while the arm and the CUDA
    major are specific to ``sgl_kernel``. More than one result is the #384
    hazard for ANY package: pip does not treat it as a conflict, so the last
    install silently wins.
    """
    sp = Path(site_packages) if site_packages is not None else _default_site_packages()
    out: List[ProviderDist] = []
    if not sp.is_dir():
        return out
    for dist_info in sorted(sp.glob("*.dist-info")):
        if not dist_info.is_dir():
            continue
        recorded = _record_provides_package(dist_info, package)
        if recorded == 0:
            continue
        name, version = _read_metadata(dist_info)
        url, sha = _read_direct_url(dist_info)
        out.append(
            ProviderDist(
                dist_name=name,
                version=version,
                dist_info=dist_info,
                recorded_files=recorded,
                direct_url=url,
                direct_url_sha256=sha,
            )
        )
    return out


def _default_site_packages() -> Path:
    purelib = sysconfig.get_paths().get("purelib")
    return Path(purelib) if purelib else Path(sys.prefix)


def inspect_sgl_kernel(
    site_packages: Optional[Path] = None,
    *,
    expect_sha256: Optional[str] = PINNED_WHEEL_SHA256,
) -> KernelDistReport:
    """Inspect an installation for the #384 shadow state.

    Args:
        site_packages: Directory to inspect. Defaults to the running
            interpreter's ``purelib``, which is what a preflight check wants;
            tests and the Docker layer pass an explicit path.
        expect_sha256: Wheel hash the winning distribution is expected to have
            been installed from, compared against ``direct_url.json``. Pass
            ``None`` to skip -- appropriate when the caller deliberately allows
            the stock pypi wheel, which has no ``direct_url.json`` at all.

    Returns:
        A :class:`KernelDistReport`. This function never raises for a bad
        installation state -- a bad state is a *verdict*, and the caller
        decides the policy. It raises only for an unreadable filesystem.
    """
    sp = Path(site_packages) if site_packages is not None else _default_site_packages()
    report = KernelDistReport(verdict=VERDICT_MISSING, site_packages=sp)

    if not sp.is_dir():
        report.notes.append(f"site-packages directory does not exist: {sp}")
        return report

    report.providers = list_providers(IMPORT_PACKAGE, sp)

    pkg_dir = sp / IMPORT_PACKAGE
    if pkg_dir.is_dir():
        report.package_dir = pkg_dir

    if len(report.providers) > 1:
        report.verdict = VERDICT_SHADOWED
    elif not report.providers and report.package_dir is None:
        report.verdict = VERDICT_MISSING
        report.notes.append(
            f"no installed distribution records {IMPORT_PACKAGE}/ files, and "
            f"{pkg_dir} does not exist"
        )
        return report

    # Files on disk are what actually gets imported, so the arm and the CUDA
    # major are read from them -- not from whichever RECORD happens to be
    # listed first. Under a shadow, the two disagree by construction, and the
    # disk is the side that decides what the server does.
    if report.package_dir is not None:
        objects = sorted(report.package_dir.rglob("*.so"))
        for so in objects:
            try:
                blob = so.read_bytes()
            except OSError as exc:  # pragma: no cover - unreadable file
                report.notes.append(f"could not read {so.name}: {exc}")
                continue
            # Keyed by path relative to the package dir, not by bare filename:
            # the wheel ships several objects called common_ops.abi3.so in
            # per-arch subdirectories (sm100/, ...), and a bare-name key would
            # silently collapse them into one entry.
            rel = so.relative_to(report.package_dir).as_posix()
            hits = blob.count(INT8_ARM_SYMBOL)
            if hits:
                report.arm_scan[rel] = hits
                report.arm_present = True
            try:
                majors = cuda_majors(elf_needed(so))
            except (ElfError, OSError) as exc:
                report.notes.append(f"ELF probe failed on {rel}: {exc}")
                continue
            for lib, major in majors.items():
                previous = report.kernel_cuda_majors.get(lib)
                if previous is not None and previous != major:
                    report.notes.append(
                        f"{rel} links {lib}.so.{major} while a sibling object "
                        f"links .so.{previous}"
                    )
                report.kernel_cuda_majors[lib] = major
        if not objects:
            report.notes.append(f"{report.package_dir} contains no .so objects")

    if report.verdict != VERDICT_SHADOWED:
        report.verdict = VERDICT_ARMED if report.arm_present else VERDICT_ARMLESS

    torch_major, torch_notes = _detect_torch_cuda_major(sp)
    report.torch_cuda_major = torch_major
    report.notes.extend(torch_notes)
    kernel_rt = report.kernel_cuda_majors.get("libcudart")
    if torch_major is not None and kernel_rt is not None and torch_major != kernel_rt:
        report.cuda_major_split = True
        report.notes.append(
            f"CUDA major split (#436): {IMPORT_PACKAGE} objects link "
            f"libcudart.so.{kernel_rt}, torch links libcudart.so.{torch_major}"
        )

    if expect_sha256:
        winners = [p for p in report.providers if not p.is_pypi_shadow]
        pool = winners or report.providers
        observed = {p.direct_url_sha256 for p in pool if p.direct_url_sha256}
        if not observed:
            report.sha256_mismatch = (
                "no direct_url.json hash on the providing distribution -- the "
                "pinned fork wheel is always a file:// install, so an index "
                "install is by itself off-pin"
            )
        elif expect_sha256 not in observed:
            report.sha256_mismatch = (
                f"installed from sha256 {sorted(observed)}, pin expects {expect_sha256}"
            )

    return report


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file's sha256, for verifying a wheel before it is installed."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(report: KernelDistReport, *, require_arm: bool) -> str:
    """Render a loud, self-explaining report.

    Loud is the requirement, not a style choice: the whole failure mode of #384
    is that it is silent. Every line names what was measured and where, so the
    reader can re-run the measurement by hand.
    """
    lines: List[str] = []
    lines.append(f"sgl_kernel dist guard (#384): verdict={report.verdict}")
    lines.append(f"  site-packages: {report.site_packages}")
    if not report.providers:
        lines.append("  providing distributions: NONE")
    for prov in report.providers:
        tag = "  <-- armless pypi dist" if prov.is_pypi_shadow else ""
        lines.append(
            f"  provider: {prov.dist_name} {prov.version} "
            f"({prov.recorded_files} recorded {IMPORT_PACKAGE}/ files){tag}"
        )
        lines.append(f"    dist-info: {prov.dist_info.name}")
        if prov.direct_url:
            lines.append(f"    installed from: {prov.direct_url}")
        if prov.direct_url_sha256:
            lines.append(f"    sha256: {prov.direct_url_sha256}")
    lines.append(f"  package dir: {report.package_dir}")
    lines.append(
        f"  INT8 arm ({INT8_ARM_SYMBOL.decode()}): "
        f"{'present' if report.arm_present else 'ABSENT'}"
        + (f" in {report.arm_scan}" if report.arm_scan else "")
    )
    lines.append(
        f"  CUDA majors: {IMPORT_PACKAGE}={report.kernel_cuda_majors or '?'} "
        f"torch=libcudart.so.{report.torch_cuda_major}"
    )
    for note in report.notes:
        lines.append(f"  note: {note}")

    problems = describe_problems(report, require_arm=require_arm)
    if problems:
        lines.append("")
        lines.append("PROBLEMS:")
        for problem in problems:
            lines.append(f"  * {problem}")
        lines.append("")
        # .../<venv>/lib/python3.X/site-packages -> <venv>
        venv = report.site_packages.parent.parent.parent
        lines.append(REPAIR_RECIPE.format(venv=venv))
    return "\n".join(lines)


def describe_problems(report: KernelDistReport, *, require_arm: bool) -> List[str]:
    """Every reason this installation is not acceptable, in plain words."""
    problems: List[str] = []
    if report.verdict == VERDICT_SHADOWED:
        names = ", ".join(f"{p.dist_name} {p.version}" for p in report.providers)
        problems.append(
            f"WHEEL SHADOW (#384): {len(report.providers)} distributions provide "
            f"the same {IMPORT_PACKAGE} import package ({names}). pip sees no "
            f"conflict between them, so the next 'pip install' of either one "
            f"silently decides which files win. This is an error even though "
            f"the arm is currently "
            f"{'present' if report.arm_present else 'absent'}."
        )
    if report.verdict == VERDICT_MISSING:
        problems.append(
            f"{IMPORT_PACKAGE} is not installed at all in {report.site_packages}."
        )
    if require_arm and not report.arm_present:
        problems.append(
            f"INT8 arm missing: {INT8_ARM_SYMBOL.decode()} was not found in any "
            f"object under {report.package_dir}. The INT8-W8A8 production "
            f"default cannot serve without it; the boot fails during layer "
            f"construction inside the JIT cold-build window."
        )
    if report.cuda_major_split:
        problems.append(
            f"CUDA major split (#436): the wheel links "
            f"libcudart.so.{report.kernel_cuda_majors.get('libcudart')} but torch "
            f"links libcudart.so.{report.torch_cuda_major}. cudaRuntimeGetVersion "
            f"binds to the wheel's cudart while dlsym binds to torch's, so the "
            f"two disagree about the cudaMemcpyBatchAsync signature -- a "
            f"deterministic SIGSEGV in the HiCache host tier, not a race."
        )
    if report.sha256_mismatch:
        problems.append(f"off-pin wheel: {report.sha256_mismatch}")
    return problems


# ---------------------------------------------------------------------------
# CLI -- the Docker build-time RUN layer and the host preflight both use this
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kernel_dist_guard",
        description=(
            "Detect the #384 sgl_kernel wheel-shadow state by file inspection. "
            "Exits non-zero when the installation is unacceptable under the "
            "policy given by the flags."
        ),
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=None,
        help="directory to inspect (default: this interpreter's purelib)",
    )
    parser.add_argument(
        "--require-arm",
        action="store_true",
        help=(
            "fail when the INT8 arm is absent. Leave unset for a deliberately "
            "armless build (stock pypi wheel / Turing-only image), which is a "
            "supported configuration -- the shadow check still applies."
        ),
    )
    parser.add_argument(
        "--expect-sha256",
        default=None,
        help=(
            "wheel sha256 the providing dist must have been installed from. "
            "Use --expect-pinned-sha256 for the runbook 2.1 pin."
        ),
    )
    parser.add_argument(
        "--expect-pinned-sha256",
        action="store_true",
        help=f"shorthand for --expect-sha256 {PINNED_WHEEL_SHA256}",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help=(
            "verify this wheel FILE's sha256 before anything is installed, and "
            "exit. Used by the Docker layer to refuse an off-pin build input."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON on stdout"
    )
    args = parser.parse_args(argv)

    if args.wheel is not None:
        expected = args.expect_sha256 or (
            PINNED_WHEEL_SHA256 if args.expect_pinned_sha256 else None
        )
        if not args.wheel.is_file():
            print(f"FATAL: wheel not found: {args.wheel}", file=sys.stderr)
            return 2
        observed = sha256_file(args.wheel)
        print(f"wheel: {args.wheel}\nsha256: {observed}")
        if expected and observed != expected:
            print(
                f"FATAL: wheel sha256 {observed} does not match the pin {expected}. "
                f"Refusing to build an image from an unpinned kernel wheel; see "
                f"docs/rig-runbook.md 2.1 for the wheel provenance table.",
                file=sys.stderr,
            )
            return 1
        return 0

    expect = args.expect_sha256
    if args.expect_pinned_sha256 and not expect:
        expect = PINNED_WHEEL_SHA256

    report = inspect_sgl_kernel(args.site_packages, expect_sha256=expect)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report, require_arm=args.require_arm))

    problems = describe_problems(report, require_arm=args.require_arm)
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    raise SystemExit(main())
