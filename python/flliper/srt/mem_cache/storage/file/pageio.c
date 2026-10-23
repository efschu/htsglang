/* #1402: batched page I/O for the HiCache file backend.
 *
 * WHY C. Every os.* call in Python releases and re-acquires the GIL, and in a
 * scheduler process whose main thread is busy the re-acquire waits up to the
 * interpreter's switch interval (5 ms by default). Measured 2026-09-15 against
 * boot xsn133's store: a page read of six syscalls costs 13 us alone and
 * 20.6 ms beside a busy Python thread. One ctypes call releases the GIL ONCE
 * for a whole batch, so 128 pages cost one hand-off instead of ~800.
 *
 * Both entry points are plain loops over plain syscalls; nothing here decides
 * anything about keys, windows or readability -- that stays in Python.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* sizes[i] = st_size of paths[i], or -1 when it cannot be stat'ed.
 * Returns the number of paths found. */
int64_t hicache_stat_sizes(int64_t n, const char **paths, int64_t *sizes) {
    struct stat st;
    int64_t found = 0;
    for (int64_t i = 0; i < n; i++) {
        if (stat(paths[i], &st) == 0) {
            sizes[i] = (int64_t)st.st_size;
            found++;
        } else {
            sizes[i] = -1;
        }
    }
    return found;
}

/* Read the extents of n canonical blobs into n caller buffers.
 *
 * For page i: expect_total[i] >= 0 demands that exact file size (a blob of
 * another width is refused, status 2); n_ext[i] extents follow in ext_off /
 * ext_len (flattened over all pages, in page order) and land back to back in
 * out[i]. touch != 0 bumps the mtime through the open fd (the LRU recency the
 * evictor's sibling owner reads). status[i]: 0 ok, 1 missing, 2 size
 * mismatch, 3 short read, 4 other error. Returns the number of ok pages. */
static int8_t pio_read_one(const char *path, int64_t expect_total, int64_t k,
                           const int64_t *ext_off, const int64_t *ext_len, uint8_t *out,
                           int32_t touch) {
    int8_t s = 0;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return (errno == ENOENT) ? 1 : 4;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        s = 4;
    } else if (expect_total >= 0 && (int64_t)st.st_size != expect_total) {
        s = 2;
    } else {
        int64_t taken = 0;
        for (int64_t j = 0; j < k && s == 0; j++) {
            int64_t off = ext_off[j], len = ext_len[j], got = 0;
            while (got < len) {
                ssize_t r = pread(fd, out + taken + got, (size_t)(len - got),
                                  (off_t)(off + got));
                if (r < 0) {
                    if (errno == EINTR) continue;
                    s = 4;
                    break;
                }
                if (r == 0) {
                    s = 3;
                    break;
                }
                got += r;
            }
            taken += len;
        }
        if (s == 0 && touch) {
            struct timespec ts[2] = {{0, UTIME_NOW}, {0, UTIME_NOW}};
            (void)futimens(fd, ts);
        }
    }
    close(fd);
    return s;
}

int64_t hicache_read_pages(int64_t n, const char **paths, const int64_t *expect_total,
                           const int64_t *n_ext, const int64_t *ext_off,
                           const int64_t *ext_len, uint8_t **out, int32_t touch,
                           int8_t *status) {
    int64_t ok = 0, e = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t k = n_ext[i];
        int8_t s = pio_read_one(paths[i], expect_total[i], k, ext_off + e, ext_len + e, out[i], touch);
        status[i] = s;
        if (s == 0) ok++;
        e += k;
    }
    return ok;
}

/* ---------------------------------------------------------------------------
 * STOREREAD-1023: the same page reads on nthreads C threads in ONE call.
 *
 * Boot 15 of the 27B duo line (1023_114625, P PP0 11:51:17-37Z): L3 -> L2
 * fills of 1024 pages took 3.2-5.1 s (200-316 pages/s, 6-10 MB/s) while the
 * disk gives 38.7k pages/s cold to the same reader (storeread probe, 1023).
 * Every return from a ctypes call into Python re-acquires the GIL; beside a
 * busy Python thread (the scheduler loop, the background L3 evictor) each
 * re-acquire waits up to the switch interval. The pool path returned to
 * Python after every 16 pages on each of 16 threads (plus a per-page Python
 * os.stat in the caller). Here the whole batch is read by C threads that pull
 * page indices from one atomic counter; Python is re-entered once. ``done``
 * (optional) counts finished pages, so a Python caller can report the read's
 * progress (the S23 JOIN-IDLE witness) only when bytes actually moved.
 * ------------------------------------------------------------------------- */
#include <pthread.h>

typedef struct {
    int64_t n;
    const char **paths;
    const int64_t *expect_total;
    const int64_t *n_ext;
    const int64_t *ext_start;
    const int64_t *ext_off;
    const int64_t *ext_len;
    uint8_t **out;
    int32_t touch;
    int8_t *status;
    int64_t next;
    int64_t ok;
    int64_t *done;
} pio_mt_job;

static void *pio_mt_worker(void *arg) {
    pio_mt_job *j = (pio_mt_job *)arg;
    int64_t ok = 0;
    for (;;) {
        int64_t i = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (i >= j->n) break;
        int64_t e = j->ext_start[i];
        int8_t s = pio_read_one(j->paths[i], j->expect_total[i], j->n_ext[i], j->ext_off + e,
                                j->ext_len + e, j->out[i], j->touch);
        j->status[i] = s;
        if (s == 0) ok++;
        if (j->done) __atomic_fetch_add(j->done, 1, __ATOMIC_RELAXED);
    }
    __atomic_fetch_add(&j->ok, ok, __ATOMIC_RELAXED);
    return NULL;
}

int64_t hicache_read_pages_mt(int64_t n, const char **paths, const int64_t *expect_total,
                              const int64_t *n_ext, const int64_t *ext_off,
                              const int64_t *ext_len, uint8_t **out, int32_t touch,
                              int8_t *status, int32_t nthreads, int64_t *done) {
    if (n <= 0) return 0;
    int64_t *ext_start = (int64_t *)malloc(sizeof(int64_t) * (size_t)n);
    if (!ext_start) return hicache_read_pages(n, paths, expect_total, n_ext, ext_off, ext_len, out, touch, status);
    int64_t e = 0;
    for (int64_t i = 0; i < n; i++) {
        ext_start[i] = e;
        e += n_ext[i];
        status[i] = 4;
    }
    pio_mt_job job = {n, paths, expect_total, n_ext, ext_start, ext_off, ext_len, out, touch, status, 0, 0, done};
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    if ((int64_t)nthreads > n) nthreads = (int32_t)n;
    pthread_t tids[64];
    int started = 0;
    for (int t = 1; t < nthreads; t++) {
        if (pthread_create(&tids[started], NULL, pio_mt_worker, &job) == 0) started++;
    }
    pio_mt_worker(&job); /* the calling thread reads too; a failed create only costs parallelism */
    for (int t = 0; t < started; t++) pthread_join(tids[t], NULL);
    free(ext_start);
    return job.ok;
}

/* ---------------------------------------------------------------------------
 * Batched canonical extent WRITE: the same protocol as canonical_page_store's
 * write_extents / _open_part_locked, page by page, under one GIL release.
 *
 *   final exists                    -> status 2 (already complete, untouched)
 *   open(part, O_RDWR|O_CREAT) + flock(EX); final exists now -> 2; the fd's
 *     inode must be the path's inode (a writer that renamed it away between
 *     open and lock is detected and the open retried, 8 attempts -> 4)
 *   part not empty: read the marker sidecar and decode it; a missing or
 *     foreign marker (wrong magic/total/bounds) resets the part to 0 bytes
 *   ftruncate(part, total) when its size differs; pwrite each extent
 *   coverage = merge(recorded + written); not full -> write marker -> 1
 *   full -> optional fsync, rename(part, final), unlink(marker) -> 0
 *
 * Marker format (Python: _MARKER_MAGIC + struct "<QH" + count * "<QQ"):
 *   "SGL706\x02" | u64 total | u16 count | count x (u64 lo, u64 hi), LE.
 * ------------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>

#define PIO_MAGIC "SGL706\x02"
#define PIO_MAGIC_LEN 7
#define PIO_MAX_IVALS 4096

typedef struct { int64_t lo, hi; } pio_ival;

static int pio_cmp(const void *a, const void *b) {
    int64_t x = ((const pio_ival *)a)->lo, y = ((const pio_ival *)b)->lo;
    return (x > y) - (x < y);
}

/* sort + merge in place; returns the merged count */
static int64_t pio_merge(pio_ival *v, int64_t n) {
    if (n <= 1) return n;
    qsort(v, (size_t)n, sizeof(pio_ival), pio_cmp);
    int64_t w = 0;
    for (int64_t i = 1; i < n; i++) {
        if (v[i].lo <= v[w].hi) {
            if (v[i].hi > v[w].hi) v[w].hi = v[i].hi;
        } else {
            v[++w] = v[i];
        }
    }
    return w + 1;
}

static void pio_put_u64(uint8_t *p, uint64_t x) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(x >> (8 * i));
}
static uint64_t pio_get_u64(const uint8_t *p) {
    uint64_t x = 0;
    for (int i = 0; i < 8; i++) x |= (uint64_t)p[i] << (8 * i);
    return x;
}

/* decode the marker at `path` into v (capacity cap); -1 = unusable/missing */
static int64_t pio_read_marker(const char *path, int64_t total, pio_ival *v, int64_t cap) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    size_t max = (size_t)(PIO_MAGIC_LEN + 10 + 16 * cap);
    uint8_t *buf = (uint8_t *)malloc(max + 1);
    if (!buf) { close(fd); return -1; }
    ssize_t got = 0, r;
    while ((size_t)got < max + 1) {
        r = read(fd, buf + got, max + 1 - (size_t)got);
        if (r < 0) { if (errno == EINTR) continue; got = -1; break; }
        if (r == 0) break;
        got += r;
    }
    close(fd);
    int64_t n = -1;
    if (got >= PIO_MAGIC_LEN + 10 && memcmp(buf, PIO_MAGIC, PIO_MAGIC_LEN) == 0) {
        uint64_t rec_total = pio_get_u64(buf + PIO_MAGIC_LEN);
        uint64_t count = (uint64_t)buf[PIO_MAGIC_LEN + 8] | ((uint64_t)buf[PIO_MAGIC_LEN + 9] << 8);
        if (rec_total == (uint64_t)total && count <= (uint64_t)cap &&
            (uint64_t)got == (uint64_t)(PIO_MAGIC_LEN + 10) + 16 * count) {
            n = (int64_t)count;
            for (uint64_t i = 0; i < count; i++) {
                const uint8_t *p = buf + PIO_MAGIC_LEN + 10 + 16 * i;
                v[i].lo = (int64_t)pio_get_u64(p);
                v[i].hi = (int64_t)pio_get_u64(p + 8);
                if (!(0 <= v[i].lo && v[i].lo < v[i].hi && v[i].hi <= total)) { n = -1; break; }
            }
        }
    }
    free(buf);
    return n;
}

static int pio_write_marker(const char *path, int64_t total, const pio_ival *v, int64_t n) {
    size_t len = (size_t)(PIO_MAGIC_LEN + 10 + 16 * n);
    uint8_t *buf = (uint8_t *)malloc(len);
    if (!buf) return -1;
    memcpy(buf, PIO_MAGIC, PIO_MAGIC_LEN);
    pio_put_u64(buf + PIO_MAGIC_LEN, (uint64_t)total);
    buf[PIO_MAGIC_LEN + 8] = (uint8_t)(n & 0xff);
    buf[PIO_MAGIC_LEN + 9] = (uint8_t)((n >> 8) & 0xff);
    for (int64_t i = 0; i < n; i++) {
        pio_put_u64(buf + PIO_MAGIC_LEN + 10 + 16 * i, (uint64_t)v[i].lo);
        pio_put_u64(buf + PIO_MAGIC_LEN + 10 + 16 * i + 8, (uint64_t)v[i].hi);
    }
    int rc = 0;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) rc = -1;
    else {
        size_t done = 0;
        while (done < len) {
            ssize_t w = write(fd, buf + done, len - done);
            if (w < 0) { if (errno == EINTR) continue; rc = -1; break; }
            done += (size_t)w;
        }
        close(fd);
    }
    free(buf);
    return rc;
}

/* open the part locked; 0 ok (fd out), 2 final complete, 4 failed */
static int pio_open_part_locked(const char *part, const char *final_path, int *fd_out) {
    struct stat on_disk, mine;
    for (int attempt = 0; attempt < 8; attempt++) {
        int fd = open(part, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) return 4;
        if (flock(fd, LOCK_EX) != 0) { close(fd); return 4; }
        if (access(final_path, F_OK) == 0) { flock(fd, LOCK_UN); close(fd); return 2; }
        if (stat(part, &on_disk) != 0) { flock(fd, LOCK_UN); close(fd); continue; }
        if (fstat(fd, &mine) == 0 && on_disk.st_ino == mine.st_ino) { *fd_out = fd; return 0; }
        flock(fd, LOCK_UN);
        close(fd);
    }
    return 4;
}

/* status per page: 0 completed+published, 1 partial (marker written),
 * 2 already complete, 3 payload shape error, 4 io/lock error.
 * Returns the number of pages with status 0 or 1. */
int64_t hicache_write_pages(int64_t n, const char **finals, const char **parts,
                            const char **markers, const int64_t *totals,
                            const int64_t *n_ext, const int64_t *ext_off,
                            const int64_t *ext_len, const uint8_t **payload,
                            int32_t do_fsync, int8_t *status) {
    int64_t ok = 0, e = 0;
    pio_ival *cov = (pio_ival *)malloc(sizeof(pio_ival) * (PIO_MAX_IVALS + 64));
    if (!cov) { for (int64_t i = 0; i < n; i++) status[i] = 4; return 0; }
    for (int64_t i = 0; i < n; i++) {
        int64_t k = n_ext[i], total = totals[i];
        int8_t s = 0;
        int fd = -1;
        if (k > 64) { status[i] = 3; e += k; continue; }
        if (access(finals[i], F_OK) == 0) { status[i] = 2; e += k; continue; }
        int rc = pio_open_part_locked(parts[i], finals[i], &fd);
        if (rc != 0) { status[i] = (int8_t)rc; e += k; continue; }
        struct stat st;
        int64_t ncov = 0;
        if (fstat(fd, &st) != 0) s = 4;
        else {
            if (st.st_size != 0) {
                ncov = pio_read_marker(markers[i], total, cov, PIO_MAX_IVALS);
                if (ncov < 0) { ncov = 0; if (ftruncate(fd, 0) != 0) s = 4; st.st_size = 0; }
            }
            if (s == 0 && st.st_size != total && ftruncate(fd, total) != 0) s = 4;
        }
        int64_t taken = 0;
        for (int64_t j = 0; j < k && s == 0; j++) {
            int64_t off = ext_off[e + j], len = ext_len[e + j], done = 0;
            while (done < len) {
                ssize_t w = pwrite(fd, payload[i] + taken + done, (size_t)(len - done), (off_t)(off + done));
                if (w < 0) { if (errno == EINTR) continue; s = 4; break; }
                done += w;
            }
            taken += len;
            if (s == 0) { cov[ncov].lo = off; cov[ncov].hi = off + len; ncov++; }
        }
        if (s == 0) {
            ncov = pio_merge(cov, ncov);
            int full = (ncov == 1 && cov[0].lo == 0 && cov[0].hi == total);
            if (!full) {
                if (pio_write_marker(markers[i], total, cov, ncov) != 0) s = 4; else s = 1;
            } else {
                if (do_fsync && fsync(fd) != 0) s = 4;
                else if (rename(parts[i], finals[i]) != 0) s = 4;
                else { if (unlink(markers[i]) != 0 && errno != ENOENT) { /* best effort */ } s = 0; }
            }
        }
        flock(fd, LOCK_UN);
        close(fd);
        status[i] = s;
        if (s == 0 || s == 1) ok++;
        e += k;
    }
    free(cov);
    return ok;
}

/* ---------------------------------------------------------------------------
 * S36 (18.10.) KV-DURCHLAUF: read whole canonical pages out of file
 * descriptors the caller HOLDS (the L3 file was opened when the store read
 * resolved the page; an fd keeps the inode alive even if the L3 LRU unlinks
 * the path meanwhile -- the page is still the page its key names, writers
 * publish by rename). fds[i] may be an O_DIRECT descriptor (XFS): a pread
 * that the kernel refuses for alignment (EINVAL) is repeated through a
 * buffered descriptor of the SAME inode (/proc/self/fd/<fd>), so a misaligned
 * target never fails the page. status[i]: 0 ok, 3 short, 4 error.
 * direct_out[i]: 1 when the page went through O_DIRECT, 0 buffered.
 * ------------------------------------------------------------------------- */
static int pio_pread_full(int fd, uint8_t *dst, int64_t len, int64_t off) {
    int64_t got = 0;
    while (got < len) {
        ssize_t r = pread(fd, dst + got, (size_t)(len - got), (off_t)(off + got));
        if (r < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (r == 0) return 3;
        got += r;
    }
    return 0;
}

int64_t hicache_read_fds(int64_t n, const int32_t *fds, const int64_t *offs,
                         const int64_t *lens, uint8_t **out, int8_t *status,
                         int8_t *direct_out) {
    int64_t ok = 0;
    for (int64_t i = 0; i < n; i++) {
        int8_t s = 0, d = 0;
        int fd = fds[i];
        if (fd < 0) { status[i] = 4; direct_out[i] = 0; continue; }
        int fl = fcntl(fd, F_GETFL);
        int is_direct = (fl >= 0) && (fl & O_DIRECT);
        int rc = pio_pread_full(fd, out[i], lens[i], offs[i]);
        if (rc == -EINVAL && is_direct) {
            char p[64];
            snprintf(p, sizeof p, "/proc/self/fd/%d", fd);
            int bfd = open(p, O_RDONLY | O_CLOEXEC);
            if (bfd < 0) rc = -errno;
            else { rc = pio_pread_full(bfd, out[i], lens[i], offs[i]); close(bfd); }
        } else if (rc == 0 && is_direct) {
            d = 1;
        }
        if (rc == 3) s = 3; else if (rc != 0) s = 4;
        status[i] = s;
        direct_out[i] = d;
        if (s == 0) ok++;
    }
    return ok;
}

/* S36 Soll 2: the canonical extent WRITE of hicache_write_pages, with every
 * extent that is aligned for O_DIRECT (payload address, file offset and
 * length multiples of 4096) written through an O_DIRECT descriptor of the
 * locked part file -- the L2 -> L3 copy bypasses the page cache on XFS. An
 * extent that is not aligned, a filesystem without O_DIRECT (tmpfs: EINVAL at
 * open) or a refused pwrite goes through the buffered descriptor as before.
 * direct_out[i] counts the extents of page i that went direct. */
#define PIO_ALIGN 4096
int64_t hicache_write_pages_direct(int64_t n, const char **finals, const char **parts,
                                   const char **markers, const int64_t *totals,
                                   const int64_t *n_ext, const int64_t *ext_off,
                                   const int64_t *ext_len, const uint8_t **payload,
                                   int32_t do_fsync, int8_t *status, int32_t *direct_out) {
    int64_t ok = 0, e = 0;
    pio_ival *cov = (pio_ival *)malloc(sizeof(pio_ival) * (PIO_MAX_IVALS + 64));
    if (!cov) { for (int64_t i = 0; i < n; i++) { status[i] = 4; direct_out[i] = 0; } return 0; }
    for (int64_t i = 0; i < n; i++) {
        int64_t k = n_ext[i], total = totals[i];
        int8_t s = 0;
        int fd = -1, dfd = -1, dfd_tried = 0;
        direct_out[i] = 0;
        if (k > 64) { status[i] = 3; e += k; continue; }
        if (access(finals[i], F_OK) == 0) { status[i] = 2; e += k; continue; }
        int rc = pio_open_part_locked(parts[i], finals[i], &fd);
        if (rc != 0) { status[i] = (int8_t)rc; e += k; continue; }
        struct stat st;
        int64_t ncov = 0;
        if (fstat(fd, &st) != 0) s = 4;
        else {
            if (st.st_size != 0) {
                ncov = pio_read_marker(markers[i], total, cov, PIO_MAX_IVALS);
                if (ncov < 0) { ncov = 0; if (ftruncate(fd, 0) != 0) s = 4; st.st_size = 0; }
            }
            if (s == 0 && st.st_size != total && ftruncate(fd, total) != 0) s = 4;
        }
        int64_t taken = 0;
        for (int64_t j = 0; j < k && s == 0; j++) {
            int64_t off = ext_off[e + j], len = ext_len[e + j], done = 0;
            const uint8_t *src = payload[i] + taken;
            int aligned = ((((uintptr_t)src) | (uintptr_t)off | (uintptr_t)len) % PIO_ALIGN) == 0 && len > 0;
            if (aligned && !dfd_tried) {
                dfd_tried = 1;
                dfd = open(parts[i], O_WRONLY | O_DIRECT | O_CLOEXEC);
            }
            int use = (aligned && dfd >= 0) ? dfd : fd;
            while (done < len) {
                ssize_t w = pwrite(use, src + done, (size_t)(len - done), (off_t)(off + done));
                if (w < 0) {
                    if (errno == EINTR) continue;
                    if (use == dfd && errno == EINVAL) { use = fd; continue; }
                    s = 4;
                    break;
                }
                done += w;
            }
            if (s == 0 && use == dfd) direct_out[i]++;
            taken += len;
            if (s == 0) { cov[ncov].lo = off; cov[ncov].hi = off + len; ncov++; }
        }
        if (dfd >= 0) close(dfd);
        if (s == 0) {
            ncov = pio_merge(cov, ncov);
            int full = (ncov == 1 && cov[0].lo == 0 && cov[0].hi == total);
            if (!full) {
                if (pio_write_marker(markers[i], total, cov, ncov) != 0) s = 4; else s = 1;
            } else {
                if (do_fsync && fsync(fd) != 0) s = 4;
                else if (rename(parts[i], finals[i]) != 0) s = 4;
                else { if (unlink(markers[i]) != 0 && errno != ENOENT) { /* best effort */ } s = 0; }
            }
        }
        flock(fd, LOCK_UN);
        close(fd);
        status[i] = s;
        if (s == 0 || s == 1) ok++;
        e += k;
    }
    free(cov);
    return ok;
}
