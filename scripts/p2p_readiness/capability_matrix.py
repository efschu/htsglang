#!/usr/bin/env python3
"""P2P capability matrix after the driver update.

Per ordered card pair: cudaDeviceCanAccessPeer, the raw `nvidia-smi topo -m`,
driver/NCCL versions, and the BAR1 survey -- NOMINAL BAR1 size per card
(NVML / nvidia-smi -q -d MEMORY / lspci -vv, labelled nominal: it is an upper
bound, not a usability promise) plus, unless --no-aperture, the EMPIRICALLY
usable aperture per directed pair: the largest peer-writable target region
(pattern in, read back, verified) and the largest single-copy size. Mapping
failures above some size are recorded as measurements, never as aborts.

Cards are identified by PCI bus id everywhere (torch/cudart enumeration and
NVML enumeration DIVERGE on this rig -- the standing device-order trap).

Usage:
    python capability_matrix.py --out results/<date>/capability_matrix.json
    python capability_matrix.py --dry-run      # no CUDA, prints the plan

Runtime: seconds without the aperture probe; the probe adds roughly a minute
per rig (a few dozen copies up to --aperture-limit per directed pair).

This rig (2x RTX 3080 = small 256-MiB BAR window, 1x RTX 5090 = full 32-GiB
BAR): the interesting directed cases are INTO a 3080 (through the window)
vs INTO the 5090 (through full BAR). Which collective ends up where is an
OPEN question the measurements answer -- nothing here presumes an outcome.
"""

import argparse
import ctypes
import subprocess
import sys

from p2p_common import (
    MIB,
    DirectedPairResult,
    aperture_search_plan,
    classify_bar,
    cuda_can_access_peer,
    cuda_check,
    cudart,
    join_cuda_indices,
    nvml_devices,
    parse_lspci_regions,
    parse_smi_bar1,
    result_envelope,
    write_json,
)


def run_cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001 -- absence of a tool is data
        return f"<failed: {e}>"


def gather_versions():
    out = {"driver": None, "nccl": None}
    smi = run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    out["driver"] = smi.strip().splitlines()[0] if smi and "<failed" not in smi else smi
    try:
        import torch

        out["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        out["torch"] = torch.__version__
    except Exception as e:  # noqa: BLE001
        out["nccl"] = f"<unavailable: {e}>"
    return out


def survey_bars(devs):
    smi_bar1 = parse_smi_bar1(run_cmd(["nvidia-smi", "-q", "-d", "MEMORY"]))
    lspci_by_pci = {}
    for d in devs:
        text = run_cmd(["lspci", "-vv", "-s", d.pci_bus_id])
        lspci_by_pci[d.pci_bus_id] = {
            "raw_region_sizes": parse_lspci_regions(text),
        }
        if d.bar1_total_bytes is None:
            d.bar1_total_bytes = smi_bar1.get(d.pci_bus_id) or (
                lspci_by_pci[d.pci_bus_id]["raw_region_sizes"][0]
                if lspci_by_pci[d.pci_bus_id]["raw_region_sizes"]
                else None
            )
        d.bar1_classification = classify_bar(d.bar1_total_bytes, d.vram_total_bytes)
    return lspci_by_pci


# ---------------------------------------------------------------------------
# effective aperture (ctypes cudart; pattern-verified peer writes)
# ---------------------------------------------------------------------------


def _try_peer_write(src_dev, dst_dev, size, chunk=None):
    """Fill a size-byte region on dst from src, in one piece or in chunks;
    verify content on readback. Returns (ok, error_str)."""
    lib = cudart()
    src_ptr = ctypes.c_void_p()
    dst_ptr = ctypes.c_void_p()
    try:
        cuda_check(lib.cudaSetDevice(src_dev), "set src")
        cuda_check(
            lib.cudaMalloc(ctypes.byref(src_ptr), ctypes.c_size_t(size)), "malloc src"
        )
        pattern = (src_dev * 16 + dst_dev + 1) & 0xFF
        cuda_check(
            lib.cudaMemset(src_ptr, pattern, ctypes.c_size_t(size)), "memset src"
        )
        cuda_check(lib.cudaSetDevice(dst_dev), "set dst")
        cuda_check(
            lib.cudaMalloc(ctypes.byref(dst_ptr), ctypes.c_size_t(size)), "malloc dst"
        )
        cuda_check(lib.cudaMemset(dst_ptr, 0, ctypes.c_size_t(size)), "zero dst")
        cuda_check(lib.cudaSetDevice(src_dev), "set src2")
        step = chunk or size
        off = 0
        while off < size:
            n = min(step, size - off)
            cuda_check(
                lib.cudaMemcpyPeer(
                    ctypes.c_void_p(dst_ptr.value + off),
                    dst_dev,
                    ctypes.c_void_p(src_ptr.value + off),
                    src_dev,
                    ctypes.c_size_t(n),
                ),
                f"memcpyPeer off={off}",
            )
            off += n
        cuda_check(lib.cudaDeviceSynchronize(), "sync")
        # verify: read back head, middle, tail windows (full readback of
        # GiB-sized regions is pointless; a broken aperture shows up here)
        host = ctypes.create_string_buffer(4096)
        for probe_off in {0, size // 2 & ~0xFFF, max(0, size - 4096)}:
            n = min(4096, size - probe_off)
            cuda_check(
                lib.cudaMemcpy(
                    host,
                    ctypes.c_void_p(dst_ptr.value + probe_off),
                    ctypes.c_size_t(n),
                    2,  # cudaMemcpyDeviceToHost
                ),
                "readback",
            )
            if any(b != pattern for b in host.raw[:n]):
                return False, f"content mismatch at offset {probe_off}"
        return True, None
    except RuntimeError as e:
        return False, str(e)
    finally:
        for dev, ptr in ((src_dev, src_ptr), (dst_dev, dst_ptr)):
            if ptr.value:
                lib.cudaSetDevice(dev)
                lib.cudaFree(ptr)


def _enable_peer(src_dev, dst_dev):
    lib = cudart()
    cuda_check(lib.cudaSetDevice(src_dev), "set src")
    err = lib.cudaDeviceEnablePeerAccess(dst_dev, 0)
    # 704 = cudaErrorPeerAccessAlreadyEnabled -- fine
    if err not in (0, 704):
        lib.cudaGetLastError()
        return False, f"cudaDeviceEnablePeerAccess -> {err}"
    lib.cudaGetLastError()
    return True, None


def probe_effective_aperture(pair, src_dev, dst_dev, limit_bytes):
    """Growing+binary search for the largest verified single-copy peer write
    and the largest chunked-fill region. Failures are RESULTS."""
    ok, err = _enable_peer(src_dev, dst_dev)
    if not ok:
        pair.probe_errors.append(err)
        return
    for attr, chunk in (
        ("effective_max_single_copy_bytes", None),
        ("effective_max_region_chunked_bytes", 64 * MIB),
    ):
        largest_ok, first_fail = 0, None
        while True:
            size = aperture_search_plan(largest_ok, first_fail)
            if size is None or (first_fail is None and size > limit_bytes):
                break
            good, err = _try_peer_write(src_dev, dst_dev, size, chunk=chunk)
            if good:
                largest_ok = size
            else:
                first_fail = size
                pair.probe_errors.append(f"{attr} size={size}: {err}")
        setattr(pair, attr, largest_ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="capability_matrix.json")
    ap.add_argument("--dry-run", action="store_true", help="no CUDA, print plan")
    ap.add_argument("--no-aperture", action="store_true")
    ap.add_argument(
        "--aperture-limit-mib",
        type=int,
        default=1024,
        help="upper bound for the effective-aperture search per pair",
    )
    args = ap.parse_args()

    if args.dry_run:
        print("plan: NVML device survey (PCI-keyed) -> BAR1 nominal survey")
        print("      (nvidia-smi -q -d MEMORY + lspci -vv, labelled NOMINAL)")
        print("      -> cudaDeviceCanAccessPeer per ordered pair")
        print(
            "      -> effective-aperture probe per ordered pair"
            f" (verify-writes up to {args.aperture_limit_mib} MiB,"
            " failures logged as results)"
        )
        print("      -> nvidia-smi topo -m raw + driver/NCCL versions")
        print(f"      -> JSON {args.out} + readable table on stdout")
        return 0

    devs = nvml_devices()
    lspci = survey_bars(devs)
    join_cuda_indices(devs)

    pairs = []
    for a in devs:
        for b in devs:
            if a.pci_bus_id == b.pci_bus_id:
                continue
            pair = DirectedPairResult(
                src_pci=a.pci_bus_id,
                dst_pci=b.pci_bus_id,
                dst_bar1_nominal_bytes=b.bar1_total_bytes,
                dst_bar1_classification=b.bar1_classification,
            )
            if a.cuda_index is None or b.cuda_index is None:
                pair.probe_errors.append("cuda index join failed")
            else:
                try:
                    pair.can_access_peer = cuda_can_access_peer(
                        a.cuda_index, b.cuda_index
                    )
                except RuntimeError as e:
                    pair.probe_errors.append(str(e))
                if pair.can_access_peer and not args.no_aperture:
                    probe_effective_aperture(
                        pair,
                        a.cuda_index,
                        b.cuda_index,
                        args.aperture_limit_mib * MIB,
                    )
            pairs.append(pair)

    payload = result_envelope("capability_matrix")
    payload.update(
        {
            "versions": gather_versions(),
            "devices": devs,
            "lspci": lspci,
            "topo_raw": run_cmd(["nvidia-smi", "topo", "-m"]),
            "directed_pairs": pairs,
            "notes": [
                "bar1_total_bytes is the NOMINAL upper bound; consumers must "
                "use effective_max_* only",
                "probe_errors are measurements (aperture limits), not failures",
            ],
        }
    )
    write_json(args.out, payload)

    # readable table
    by_pci = {d.pci_bus_id: d for d in devs}
    print(
        f"{'src':<14}{'dst':<14}{'peer':<6}{'dst BAR (nominal)':<20}"
        f"{'eff. single-copy':<18}{'eff. region':<14}"
    )
    for p in pairs:
        bar = (
            f"{(p.dst_bar1_nominal_bytes or 0) // MIB} MiB "
            f"[{p.dst_bar1_classification}]"
        )
        eff1 = (
            f"{(p.effective_max_single_copy_bytes or 0) // MIB} MiB"
            if p.effective_max_single_copy_bytes is not None
            else "-"
        )
        eff2 = (
            f"{(p.effective_max_region_chunked_bytes or 0) // MIB} MiB"
            if p.effective_max_region_chunked_bytes is not None
            else "-"
        )
        print(
            f"{by_pci[p.src_pci].name[-10:]:<14}{by_pci[p.dst_pci].name[-10:]:<14}"
            f"{str(p.can_access_peer):<6}{bar:<20}{eff1:<18}{eff2:<14}"
        )
        for e in p.probe_errors:
            print(f"    note: {e}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
