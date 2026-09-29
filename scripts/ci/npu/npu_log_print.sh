#!/bin/bash
set -euo pipefail

# Print log information(flliper version, commit sha, sgl-kernel-npu version, sgl-kernel-npu commit sha, npu-smi info and pip list.
npu-smi info
pip list
get_version() {
    [ -f "$1" ] && python3 -c 'import re, sys; print(sys.argv[2] + " version: v" + re.search(r"__version__\s*=\s*[\"'"'"'](.*?)[\"'"'"']", open(sys.argv[1]).read()).group(1))' "$1" "$2" 2>/dev/null || echo "$2 version: unknown"
}
get_version "./python/flliper/version.py" "flliper"
get_version "./sgl-kernel/python/sgl_kernel/version.py" "sgl_kernel"
FLLIPER_URL="https://github.com/sgl-project/sglang.git"
SGL_KERNEL_URL="https://github.com/sgl-project/sgl-kernel-npu.git"
FLLIPER_BRANCH="main"
SGL_KERNEL_BRANCH="main"
get_sha() {
    local name="$1"
    local url="$2"
    local branch="$3"
    local sha
    sha=$(git ls-remote "$url" "refs/heads/$branch" | cut -f1)
    echo "$name SHA for branch $branch: ${sha:-"Not Found"}"
}
get_sha "flliper" "$FLLIPER_URL" "$FLLIPER_BRANCH"
get_sha "sgl-kernel" "$SGL_KERNEL_URL" "$SGL_KERNEL_BRANCH"
chmod +x scripts/ci/npu/npu_log_print.sh
