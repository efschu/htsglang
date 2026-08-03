# DESIGN #456 — sparse hibernate image writes

**Status:** built, hermetically tested, measured on this box. BOOT-PENDING for
a real #89 round trip on GPU state (see §7).

**Slice:** `python/sglang/srt/model_loader/sparse_write.py` (mechanism),
one line in `model_loader/hibernate.py:park_weights_to_disk` (wiring),
`tests/hibernate/test_sparse_write_456.py` (23 tests),
`scripts/dev/456_sparse_write/` (measurement).

---

## 1. Where this comes from

#306 set out to compress cold-tier assets and refuted the codec
(`ANALYSE_306_lossless_ratio.md`; the register entry lives in
`planner/rejected.py`). One number survived the refutation. Streaming the whole
6.68 GiB #89 rank image (`image_whole.py`) decomposed its 1.1953x `zstd -3`
ratio into two mechanisms that a single blended figure hides:

| mechanism | ratio | cost |
|---|---|---|
| all-zero 4 KiB pages (221 542 of 1 752 229 = **12.64 %**) | **1.1447x** | no codec, no decompress stage |
| residual `zstd -3` gain on the non-hole part | 1.0442x | CPU on every write AND every read |

The holes are not noise in the data: a `torch.save` container of final GGUF
weights parks pre-allocated buffers in full, and those buffers were zeros in
VRAM. Removing them needs a `lseek`, not a codec.

This document is what happened when that 1.1447x was actually built and
measured rather than projected.

## 2. Mechanism, and why the format did not move

`SparseFileWriter` is a write-only file object that sits **under** whatever
container is being written. It detects all-zero pages in FILE-OFFSET space and
`lseek()`s over them instead of calling `write()`. POSIX guarantees a hole
reads back as zeros, which is exactly what those pages held.

So the answer to "offset-addressed or stream-framed?" is **neither, and it does
not matter**. The #89 image is a `torch.save` ZIP_STORED container — stream-
framed — but sparseness is applied one layer below the container, at the byte
sink. `torch.load`, `mmap`, `read()` and `sha256sum` all see the same bytes.
There is no header change, no frame type, no manifest field and no version
gate, because there is nothing for a reader to know. `HIBERNATE_VERSION` stays
at 2, and `test_restore_path_is_untouched` pins that.

Two details are load-bearing and both have a can-fail arm in the suite:

* **`ftruncate` on close.** `lseek` past EOF is not a write, so an image ending
  in zeros would be short by exactly its final zero run. Removing the
  `ftruncate` turns `test_trailing_hole_preserved` red (`4096 == 24576`).
* **The partial-page carry.** Holes are aligned in file-offset space; `write()`
  call boundaries are not (a `torch.save` record starts on 64 bytes). The
  writer holds back the trailing sub-page of each call and completes it from
  the next, so the hole map is identical no matter how the caller chunks.
  Without the carry, a caller writing 4099-byte pieces punches **one** page in
  the whole image instead of eight
  (`test_hole_map_is_independent_of_chunking`).

The dense escape path writes through a file OBJECT rather than the path, because
torch derives the ZIP archive name from the path it is handed — the two
containers otherwise differ in length (measured: 82 bytes on a 16 MB payload).
Routing both branches through the buffer API makes
`SGLANG_HIBERNATE_DENSE_WRITE=1` **bit-identical** to the default, which is what
lets the dense-vs-sparse sha256 comparison be a real gate.

## 3. Detector cost — the number the value claim turns on

The detector is a second full pass over the image, so its cost is not
incidental; it decides whether skipping the write is a win at all. Measured on
this box, 1 GiB, best of 4:

| variant | ms / GiB |
|---|---|
| uint8 `any(axis=1)` | 63.9 |
| **uint64 `bitwise_or.reduce(axis=1)`** (shipped) | **56.5** |
| sampled pre-filter, first word per page | 62.1 |
| sampled pre-filter, first + middle word | 63.2 |
| sampled pre-filter, contiguous copy of the first column | 57.0 |

The sampling idea — one non-zero probe proves a page non-zero without reading
the rest, and 87 % of pages are non-zero — **does not pay**. The hardware
prefetcher streams the intervening cache lines regardless, so the strided read
costs what the sequential one costs and the second gather stage is pure
overhead. The simple reduction wins. In the writer, chunked as it really runs,
the detector measures **67 ms/GiB** (0.201 s over 3 GiB, three trials within
3 ms).

The task brief called this "zero CPU". It is not zero: it is ~0.45 s for a
6.68 GiB rank image, once, at park time, and never on the restore path.

## 4. What it buys, measured, on this box

`scripts/dev/456_sparse_write/measure_sparse_write.py`: a 3 GiB synthetic image
carrying exactly the #306 hole fraction (0.126434, twelve large contiguous zero
regions, incompressible random filler at the measured 7.38 bits/byte), written
dense and sparse. Three interleaved arms — two IDENTICAL dense writes as the
A-vs-A floor plus the sparse one — arm order rotating every rep, `os.sync()` and
a 2 s drain before each timed window, `fsync` inside it. 11 reps per arm per
filesystem, medians reported (`results/measure.json`).

Two methodology notes, both of which changed the answer:

* The first version kept a fixed arm order and produced an 11 % A-vs-A floor
  between two identical writes. That was a position effect, not noise.
* Without the drain, ZFS lands the previous arm's transaction group inside the
  next arm's window: three untimed trials of the same dense write spread
  1.33–2.43 s, and the sparse arm looked 1.44x faster.

| | `/spinning` (ZFS, compression on) | `/dev/shm` (tmpfs, no compression) |
|---|---|---|
| apparent bytes | 3 221 225 472 | 3 221 225 472 |
| allocated, dense | 2 816 098 816 | 3 221 225 472 |
| allocated, sparse | **2 816 098 816** | **2 813 952 000** |
| **alloc ratio, sparse vs dense** | **1.0000** | **1.1447** |
| ratio the filesystem already delivered by itself | 1.1439 | 1.0000 |
| dense write, median s | 1.059 | 0.773 |
| sparse write, median s | 1.181 | 0.905 |
| **time speedup** | **0.897** | **0.855** |
| A-vs-A floor (identical dense arms) | 10.3 % | 16.9 % |
| dense and sparse images byte-identical | yes | yes |

### 4.1 Reading that honestly

**Disk bytes: nothing, here.** ZFS with compression on had already collapsed
every zero block — 2 816 098 816 allocated bytes with the sparse write and
2 816 098 816 without it, to the byte. The 1.1439x the filesystem returns on
its own is the same 12.64 % of zero pages, taken by a different layer. This
reproduces #306 §6.2's finding from the other direction: there the ZFS dataset
was already returning 1.1735x of `zstd -3`'s 1.1953x on the real image; here the
sparse write's entire byte win is inside what ZFS already holds.

**Disk bytes elsewhere: exactly the projection.** On tmpfs, which folds nothing,
sparse writing returns **1.1447x** — the #306 §J projection to four decimals,
now measured rather than derived.

**Write time: no win, on either filesystem.** The point estimates are
0.897 (ZFS) and 0.855 (tmpfs) — both negative — and both sit at or inside their
A-vs-A floor (10.3 % / 16.9 %). The honest reading is therefore: *no write-time
effect is established in either direction, and certainly no win.* An earlier
9-rep run of the same script (pre-carry writer) gave 1.022 / 0.915 with floors
of 10.7 % / 5.3 %; the ZFS point estimate straddling unity across two runs is
itself the result. What IS established, because it was measured directly rather
than differenced, is the detector's cost: **67 ms/GiB**, 0.201 s over 3 GiB with
3 ms of spread over three trials.

**So the task's premise is wrong on both halves, and it is worth saying plainly:**
the win is not "1.145x smaller on-disk + faster writes, zero CPU". On this box
it is *no smaller on disk, no faster, and ~0.45 s of CPU per 6.68 GiB park*.
What is real is that the image becomes sparse for a bounded, one-off cost, and
that pays on any target whose filesystem does not already fold zeros — ext4,
xfs, and every network or removable target a hibernate directory might be
pointed at.

### 4.2 When the write-time win exists

The break-even is arithmetic: skipping *f* of the bytes pays when the write path
costs more per skipped byte than the detector's 67 ms/GiB (≈ 15 GiB/s). Every
real storage device is far below that, so on a filesystem that does NOT fold
zeros the time win should approach the full 12.6 %.

Neither arm measured here is that case, and it is worth being explicit about
why rather than extrapolating:

* on ZFS the dense arm **never pays for those bytes either** — the filesystem
  folds them — so the comparison is the detector's cost against a saving that
  had already been made. The two effects are the same 12.64 %, counted once;
* on tmpfs the write is a RAM copy at roughly the same order as the detector's
  own pass, so skipping it is close to a wash by construction.

An ext4 or xfs target on a real device is the configuration where the win
should be there to take, and **this box has no such filesystem to test on** —
both `/` and `/tmp` are ZFS, `/dev/shm` is tmpfs. That arm is therefore
UNMEASURED, not measured-and-positive, and nothing in this document should be
read as claiming it.

## 5. Default decision

**Sparse is the DEFAULT.** `SGLANG_HIBERNATE_DENSE_WRITE=1` forces the old
dense write.

Default-on is justified by the cost/benefit as measured, not by the projection:

* the change is safe by construction — the images are byte-identical, proven by
  sha256 on read-back and by a tensor-by-tensor comparison of the restored
  state;
* the cost is bounded, one-off, at park time only, and directly measured rather
  than differenced: 67 ms/GiB, ~0.45 s for a 6.68 GiB rank image. It never
  appears on the restore path, which is the latency #89 exists to cut;
* the benefit is 1.1447x of allocated bytes on any target that does not already
  fold zeros;
* nothing on the restore path changes at all.

**And the case against, stated because it is real:** on THIS box the benefit is
zero — `/spinning` is ZFS with compression and has already folded the same
blocks — so the default costs ~0.45 s per park here and returns nothing. That is
why the runbook entry names the compressing-filesystem case explicitly: if your
`hibernate_dir` is on ZFS or btrfs with compression on, setting
`SGLANG_HIBERNATE_DENSE_WRITE=1` buys that time back at no cost. The default is
chosen for the general deployment, not for this rig.

It is deliberately NOT a runtime worth-it autocheck (`DESIGN_363` §20.1). An
autocheck would have to decide whether the hibernate directory's filesystem
folds zeros, and the only reliable probe for that is a write plus a settle —
ZFS accounts allocation at txg sync, not at `fsync` return, and the first
version of the measurement script read 0.30 GB for a file that settled at
2.82 GB. Paying seconds at every park to decide whether to save a fraction of a
second is the wrong trade; a documented env is.

## 6. Second consumers

`hicache_migrate.execute_plan` (`mem_cache/hicache_migrate.py:826`) is the other
raw-byte image writer in the tree — the #297 KV-resharding materializer, which
concatenates source extents into target files. It does **not** share the
hibernate writer's code, so it does not get this for free. It is a clean
follow-up: swap its `open(target, "wb")` for a `SparseFileWriter` and its
whole-file `shutil.copyfile` fast path keeps whatever sparseness the source
already had. Not done here — `execute_plan` has its own byte-permutation gate
(`verify_plan`) that would need the same can-fail treatment, and that is a
separate slice.

The module lives in `model_loader/` but imports nothing from it, so a consumer
outside the loader can adopt it without pulling the loader in.

## 7. BOOT-PENDING

Everything above is hermetic (`CUDA_VISIBLE_DEVICES=99`, synthetic payloads,
CPU only). The remaining proof is a real #89 round trip: park a live rank's GPU
state, restore it, and let the existing per-rank `byte_hash` gate in
`restore_model_from_disk` accept it. That gate is already the right instrument —
it recomputes the hash over the restored parameters and refuses to serve on a
mismatch — so the boot check is a single observation, not a new harness:

```
# In the existing hibernate recipe, after a park + restart cycle:
#   1. park with the default (sparse) writer, then restore. The restore log
#      must print "byte-hash OK". The park log must print the new
#      "#456 sparse write skipped N of M 4 KiB pages" line. Do NOT gate on a
#      percentage -- see "the hole fraction is a property of the image" below.
#   2. re-park with SGLANG_HIBERNATE_DENSE_WRITE=1 and compare:
#      sha256sum <hibernate_dir>/rank*_GPU-*.pt   # must match the sparse run
#      stat -c '%s %b' <hibernate_dir>/rank*_GPU-*.pt   # apparent size equal
```

**The hole fraction is a property of the IMAGE, not of the mechanism (#519).**
The 12.64 % above is one measurement of one file: `ANALYSE_306_lossless_ratio.md`
§J, 221 542 of 1 752 229 all-zero 4 KiB pages in a single 7.18 GB hibernate
image. The synthetic fixture in §4 reproduces exactly that fraction (0.126434)
BECAUSE it was built to -- it is the probe image's number, carried forward, not
an independent confirmation of it. The window-4 run on a different real image
reported **5.49 %** skipped, which is not a regression and not a contradiction:
holes come from padded/aligned parameter regions, so the fraction moves with
model geometry, quant mix, TP shard width and which tensors a rank happens to
hold. Two consequences:

* **No acceptance criterion may be a percentage.** The gate is the byte-hash on
  restore plus the identical sha256 across the dense and sparse writers; those
  are properties of the mechanism. A skipped-page share is an observation to
  record, and the log line already prints the measured value
  (`model_loader/hibernate.py:483`), so nothing has to be assumed.
* **A saving quoted from one image is a single-sample claim.** Any figure
  derived from 12.64 % -- allocated-bytes, wall-clock-per-park -- is as narrow
  as that one file, and should say which image it came from.

Step 2 is the one that matters: identical sha256 across the two writers on a
REAL image, not a synthetic one, is the whole safety claim. On this box's ZFS
pool `%b` is expected to be equal too — see §4.1 — and that is not a failure.
