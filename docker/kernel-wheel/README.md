# Build-context slot for locally built wheels

This directory is `COPY`ed into the image unconditionally, so it must exist
even when it is empty. That is the whole reason the README is here: a `COPY`
of a glob that matches nothing fails the build, while a `COPY` of a directory
containing only this file succeeds and produces an empty slot.

## What goes here

The fork's `sglang-kernel` wheel, when building an image that must serve
INT8-W8A8 checkpoints.

The stock image installs `sgl-kernel` from pypi, which does **not** carry the
`int8_scaled_mm` arm (`docker/htsglang.Dockerfile`, the DELIBERATE GAP note in
step 2, and #353). Building the kernel tree inside the image would add a full
CUDA toolchain and roughly 45 minutes per image for one branch, so the
supported route is a rig-local build dropped in here.

```bash
cp /spinning/wt-398-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl \
   docker/kernel-wheel/

docker build -f docker/htsglang.Dockerfile \
  --build-arg SGL_KERNEL_WHEEL=sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl \
  --build-arg REQUIRE_INT8_ARM=1 \
  -t htsglang:cu130-nccl2307-int8 .
```

`SGL_KERNEL_WHEEL` is a **bare filename**, not a path: it is resolved inside
the image against the copied slot, so a host path would not exist there.

## Why the build asserts instead of trusting the copy

Two distributions provide the same `sgl_kernel` import package and pip does
not treat that as a conflict, so an image can end up armless with nothing in
the build log marking it. `docs/rig-runbook.md` section 2.1 has the full
provenance and the hazard. The build-time guard refuses three states:

* the wheel dropped here is not the pinned one (sha256, checked before it is
  installed),
* more than one distribution ends up providing `sgl_kernel`,
* `REQUIRE_INT8_ARM=1` was requested and the arm is not in the objects.

## Hygiene

Wheels are **not** committed. `.gitignore` in this directory keeps everything
except itself and this README, so a wheel left here after a build does not
follow you into a commit — but it does still enter the build context, so
delete it when you are done to keep context uploads small.
