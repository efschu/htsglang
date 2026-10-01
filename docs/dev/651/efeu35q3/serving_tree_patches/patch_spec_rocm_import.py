"""efeu-TP14: let the serving tree use the gfx1103 build of sgl-kernel's EAGLE /
NEXTN tree ops (sgl-kernel/rocm-spec-gfx11 -> module sgl_spec_rocm) when
sgl_kernel itself is absent (the laptop has no sgl_kernel at all).

Only the IMPORT changes: if `from sgl_kernel import ...` fails on HIP, try
sgl_spec_rocm. The availability flags then turn true, the group-wide
decide_spec_kernel_backend picks "native", and the existing native dispatch
branches run unchanged. SGLANG_SPEC_ROCM_NATIVE=0 keeps the Triton path.
"""
import ast
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/efeu35q3/sglang_src/python"
P = os.path.join(ROOT, "sglang/srt/speculative/eagle_utils.py")
s = open(P).read()
assert "sgl_spec_rocm" not in s, "already patched"


def sub(old, new, label):
    global s
    assert s.count(old) == 1, f"{label}: anchor found {s.count(old)}x"
    s = s.replace(old, new, 1)


sub("""        _has_sgl_build_tree_kernel = True
    except ImportError:
        sgl_build_tree_kernel_efficient = None
""", """        _has_sgl_build_tree_kernel = True
    except ImportError:
        sgl_build_tree_kernel_efficient = None
        # efeu-TP14: gfx1103 build of the same upstream kernel (eagle_utils.cu)
        if _is_hip and os.environ.get("SGLANG_SPEC_ROCM_NATIVE", "1") != "0":
            try:
                from sgl_spec_rocm import (
                    build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
                )

                _has_sgl_build_tree_kernel = True
            except ImportError:
                sgl_build_tree_kernel_efficient = None
""", "build import")

sub("""        _has_sgl_verify_tree_greedy = True
    except ImportError:
        sgl_verify_tree_greedy = None
""", """        _has_sgl_verify_tree_greedy = True
    except ImportError:
        sgl_verify_tree_greedy = None
        if _is_hip and os.environ.get("SGLANG_SPEC_ROCM_NATIVE", "1") != "0":
            try:
                from sgl_spec_rocm import verify_tree_greedy as sgl_verify_tree_greedy

                _has_sgl_verify_tree_greedy = True
            except ImportError:
                sgl_verify_tree_greedy = None
""", "verify import")

if "\nimport os\n" not in s:
    sub("\nimport logging\n", "\nimport logging\nimport os\n", "import os")
ast.parse(s)
if "--dry" in sys.argv:
    print("dry run ok")
    sys.exit(0)
if not os.path.exists(P + ".orig-efeu-spec"):
    shutil.copy(P, P + ".orig-efeu-spec")
open(P + ".new", "w").write(s)
os.replace(P + ".new", P)
print("eagle_utils.py: sgl_spec_rocm fallback import patched (active at next start)")
