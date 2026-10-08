"""Stdlib-only fixture of the AP-A tests (Profil-Planer 06.10.): a synthetic three-card inventory (the shape of the
reference rig: NVML 0 = RTX 3080 20G, 1 = RTX 5090, 2 = RTX 3080 20G), a card-probe file with measured values for every
card, and the document the code of 173161c595 (before AP-A) assembled from it.  No GPU, no NVML."""

import json
import os

NOW = 1_790_000_000.0
U0, U1, U2 = "GPU-0000", "GPU-1111", "GPU-2222"
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "hardware_profile_ref_n3_before_apa_1006.json")


def nvml(n=3, driver="595.58"):
    spec = [
        (0, U0, "NVIDIA GeForce RTX 3080", 20480, [8, 6]),
        (1, U1, "NVIDIA GeForce RTX 5090", 32607, [12, 0]),
        (2, U2, "NVIDIA GeForce RTX 3080", 20480, [8, 6]),
    ][:n]
    cards = []
    for i, u, name, mib, cc in spec:
        cards.append({
            "nvml_index": i, "uuid": u, "name": name, "total_mib": mib, "cc": cc,
            "bar1_total_mib": 256 if cc == [8, 6] else 32768,
            "pcie_max_gen": 4 if cc == [8, 6] else 5, "pcie_max_width": 16,
            "pcie_cur_gen": 1, "pcie_cur_width": 8,
            "mem_bus_width_bits": 320 if cc == [8, 6] else 512,
            "mem_clock_max_mhz": 9501 if cc == [8, 6] else 14000,
            "sm_clock_max_mhz": 2100, "power_limit_w": 230.0, "power_default_w": 320.0,
            "pci_bus_id": f"0000:0{i}:00.0",
        })
    return cards, driver, []


def probe_card(uuid, name, sm_count=68, **kw):
    d = {
        "uuid": uuid, "name": name, "cuda_index": 0, "total_mib": 20480,
        "gemm_bf16_tflops": 60.0, "gemm_fp8_tflops": None,
        "fp8_note": "compute capability 8.6 has no fp8 tensor path (needs 8.9+)",
        "membw_read_gbs": 700.0, "membw_copy_gbs": 690.0, "membw_gemv_gbs": 650.0,
        "h2d_gbs": 6.0, "d2h_gbs": 6.5, "h2d_lat_us": 12.5, "d2h_lat_us": 14.0,
        "sm_count": sm_count, "l2_mib": 5.0, "compute_capability": "8.6",
        "gemm_int8_tflops": 180.0, "gemm_w4a8_int8_tflops": 62.0, "gemm_w4a16_tflops": 55.0,
        "lane_notes": {}, "arm_seconds": {"membw": 3.0, "bf16": 1.0},
        "sm_clock_mhz": 1900, "sm_clock_max_mhz": 2100, "temp_c": 60.0, "throttle_reasons": [],
        "seconds": 20.0,
    }
    d.update(kw)
    return d


def write_probe(dirpath, name="card_probe-ref.json", created=NOW - 3600, cards=None, driver="595.58"):
    cards = cards if cards is not None else [
        probe_card(U0, "NVIDIA GeForce RTX 3080"),
        probe_card(U1, "NVIDIA GeForce RTX 5090", sm_count=170, total_mib=32607, gemm_bf16_tflops=210.0,
                   gemm_fp8_tflops=400.0, fp8_note="", membw_read_gbs=1500.0, membw_copy_gbs=1450.0, membw_gemv_gbs=1400.0),
        probe_card(U2, "NVIDIA GeForce RTX 3080"),
    ]
    pairs = [{"src_uuid": U0, "dst_uuid": U1, "transport": "cuda p2p", "bandwidth_gbs": 11.5, "latency_us": 9.0,
              "peer_access": True, "note": ""}]
    with open(os.path.join(dirpath, name), "w") as f:
        json.dump({"version": 1, "created": created, "driver": driver, "torch_version": "2.9", "cuda_version": "13.0",
                   "cards": cards, "pairs": pairs}, f)


if __name__ == "__main__":  # python _hwprofile_fixture_1006.py <module_path> <cache_dir>: print the document of that module
    import importlib.util
    import sys
    import tempfile

    mod_path, tree_python = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location("hp_old", mod_path)
    hp = importlib.util.module_from_spec(spec)
    sys.modules["hp_old"] = hp
    spec.loader.exec_module(hp)
    ispec = importlib.util.spec_from_file_location("ci_mod", os.path.join(tree_python, "flliper", "srt", "pdflip", "card_identity.py"))
    ci = importlib.util.module_from_spec(ispec)
    sys.modules["ci_mod"] = ci
    ispec.loader.exec_module(ci)
    with tempfile.TemporaryDirectory() as d:
        write_probe(d)
        doc = hp.build(cache_dir=d, nvml=nvml(), now=NOW, identity=ci)
    doc.pop("id"), doc.pop("created")
    print(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False))
