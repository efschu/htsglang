# Release Scripts

This directory contains scripts to automate version bumping for fLLiper releases.

## Scripts

### `bump_flliper_version.py`
Updates fLLiper version across all relevant files following the pattern from [PR #10468](https://github.com/sgl-project/sglang/pull/10468).

**Usage:**
```bash
python scripts/release/bump_flliper_version.py 0.5.3rc0
```

**Files updated:**
- `Makefile`
- `benchmark/deepseek_v3/README.md`
- `docker/rocm.Dockerfile`
- `docs/get_started/install.md`
- `docs/platforms/amd_gpu.md`
- `docs/platforms/ascend_npu.md`
- `python/pyproject.toml`
- `python/pyproject_other.toml`
- `python/pyproject_npu.toml`
- `python/flliper/version.py`

### `bump_docs_install_version.py`
Bumps the release version pinned in the Mintlify install docs — both the `git clone -b v<version> ...flliper.git` "install from source" line and the version-pinned `lmsysorg/sglang:v<version>` Docker example. Mutable tags (`latest`, `dev`) are intentionally left untouched. Driven automatically on release-tag push by [`.github/workflows/bot-bump-docs-version.yml`](../../.github/workflows/bot-bump-docs-version.yml), which opens a PR with the change.

**Usage:**
```bash
python scripts/release/bump_docs_install_version.py 0.5.13
```

**Files updated:**
- `docs_new/docs/get-started/install.mdx` (Method 2: From source; Method 3: pinned Docker image)
- `docs_new/docs/hardware-platforms/amd_gpu.mdx` (Install from Source)

### `bump_kernel_version.py`
Updates the `sglang-kernel` release version across all relevant files following the pattern from [PR #10732](https://github.com/sgl-project/sglang/pull/10732).

**Usage:**
```bash
python scripts/release/bump_kernel_version.py 0.4.0
```

**Files updated:**
- `sgl-kernel/pyproject.toml`
- `sgl-kernel/pyproject_cpu.toml`
- `sgl-kernel/pyproject_rocm.toml`
- `sgl-kernel/pyproject_musa.toml`
- `sgl-kernel/python/sgl_kernel/version.py`

## Manual Testing Instructions

### Test fLLiper Version Bump

1. **Run the script:**
   ```bash
   python scripts/release/bump_flliper_version.py 0.5.4rc0
   ```

2. **Verify changes with git diff:**
   ```bash
   git diff
   ```

3. **Check specific files contain the new version:**
   ```bash
   grep -r "0.5.4rc0" python/flliper/version.py
   grep -r "0.5.4rc0" python/pyproject.toml
   grep -r "0.5.4rc0" docs/get_started/install.md
   ```

4. **Reset changes (if testing):**
   ```bash
   git checkout .
   ```

### Test Kernel Version Bump

1. **Run the script:**
   ```bash
   python scripts/release/bump_kernel_version.py 0.4.0
   ```

2. **Verify changes with git diff:**
   ```bash
   git diff
   ```

3. **Check specific files contain the new version:**
   ```bash
   grep -r "0.4.0" sgl-kernel/python/sgl_kernel/version.py
   grep -r "0.4.0" sgl-kernel/pyproject.toml
   ```

4. **Reset changes (if testing):**
   ```bash
   git checkout .
   ```

## Version Format Validation

- **fLLiper versions:** `X.Y.Z` or `X.Y.ZrcN` (e.g., `0.5.3` or `0.5.3rc0`)
- **Kernel versions:** `X.Y.Z` (e.g., `0.4.0`)

The scripts will validate the version format and exit with an error if invalid.
