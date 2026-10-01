"""efeu-TP14: standalone ROCm/gfx1103 build of sgl-kernel's EAGLE/NEXTN tree
kernels (eagle_utils.cu), same toolchain rules as rocm-gguf-gfx11:
no real-true16 (clang-21 gfx11 D16 miscompile, see that README).

    cd sgl-kernel/rocm-spec-gfx11 && python setup.py build_ext --inplace
produces _sgl_spec_rocm*.so; import sgl_spec_rocm (python wrappers).
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
AMDGPU_TARGET = os.environ.get("AMDGPU_TARGET", "gfx1100")

setup(
    name="sgl_spec_rocm",
    py_modules=["sgl_spec_rocm"],
    ext_modules=[
        CUDAExtension(
            name="_sgl_spec_rocm",
            sources=["binding.cpp", "csrc/speculative/eagle_utils.cu"],
            include_dirs=[str(ROOT / "include"), str(ROOT / "csrc")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    f"--offload-arch={AMDGPU_TARGET}",
                    "-Xclang", "-target-feature", "-Xclang", "-real-true16",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
