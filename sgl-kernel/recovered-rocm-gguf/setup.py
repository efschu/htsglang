"""Standalone ROCm/gfx1103 build of sgl-kernel's GGUF kernels.

Feasibility probe: the sources already carry USE_ROCM branches (vecdotq.cuh,
mmq.cuh, moe.cuh) and torch's HIP extension path defines -DUSE_ROCM=1, so the
question is whether the fork's exclusion of these sources from setup_rocm.py is
a genuine porting gap or only missing build wiring. This builds them alone to
find out.
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
AMDGPU_TARGET = os.environ.get("AMDGPU_TARGET", "gfx1103")

setup(
    name="gguf_rocm_probe",
    ext_modules=[
        CUDAExtension(
            name="gguf_rocm_probe",
            sources=["binding.cpp", "csrc/quantization/gguf/gguf_kernel.cu"],
            include_dirs=[str(ROOT / "include"), str(ROOT / "csrc")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    f"--offload-arch={AMDGPU_TARGET}",
                    "-DENABLE_BF16",
                    "-DENABLE_FP8",
                    "-DHIP_FP8_TYPE_E4M3=1",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
