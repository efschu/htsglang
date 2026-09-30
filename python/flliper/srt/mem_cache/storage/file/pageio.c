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
int64_t hicache_read_pages(int64_t n, const char **paths, const int64_t *expect_total,
                           const int64_t *n_ext, const int64_t *ext_off,
                           const int64_t *ext_len, uint8_t **out, int32_t touch,
                           int8_t *status) {
    int64_t ok = 0, e = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t k = n_ext[i];
        int8_t s = 0;
        int fd = open(paths[i], O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            s = (errno == ENOENT) ? 1 : 4;
        } else {
            struct stat st;
            if (fstat(fd, &st) != 0) {
                s = 4;
            } else if (expect_total[i] >= 0 && (int64_t)st.st_size != expect_total[i]) {
                s = 2;
            } else {
                int64_t taken = 0;
                for (int64_t j = 0; j < k && s == 0; j++) {
                    int64_t off = ext_off[e + j], len = ext_len[e + j], got = 0;
                    while (got < len) {
                        ssize_t r = pread(fd, out[i] + taken + got, (size_t)(len - got),
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
        }
        status[i] = s;
        if (s == 0) ok++;
        e += k;
    }
    return ok;
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
